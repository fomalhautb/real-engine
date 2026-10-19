import type { MaterialBinding, SceneBatch } from '../../../src/contracts';
import { Renderer } from '../../../src/renderer';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import type { MeshData } from '../../../src/geometry';
import type { PreparedTextureAsset } from '../../../src/texture-storage';

function texture(baseMipLevel: number, alpha = 255): PreparedTextureAsset {
  const colors = [[192, 32, 16], [32, 192, 16], [16, 32, 192], [128, 64, 32]];
  return { kind: 'prepared', format: 'rgba8', colorSpace: 'linear', width: 8, height: 8, baseMipLevel,
    levels: colors.slice(baseMipLevel).map((rgb, i) => {
      const size = 8 >> (i + baseMipLevel), data = new Uint8Array(size * size * 4);
      for (let j = 0; j < data.length; j += 4) data.set([...rgb, alpha], j);
      return { width: size, height: size, data };
    }) };
}
function quad(): MeshData {
  const vertices = new ArrayBuffer(320), v = new DataView(vertices);
  [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([x, y], i) => {
    v.setFloat32(i * 80, x, true); v.setFloat32(i * 80 + 4, y, true);
    v.setFloat32(i * 80 + 12, 1, true); v.setFloat32(i * 80 + 24, 1, true); v.setFloat32(i * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}
function binding(asset: PreparedTextureAsset, right: boolean): MaterialBinding {
  const slot = right ? 1 : 0;
  const sampler = { minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
    ...(right ? { lodMinClamp: 2, lodMaxClamp: 2 } : {}) } as const;
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', alphaMode: 'mask',
    doubleSided: true, alphaCutoff: 0.5, baseColor: m.multiply(m.texture(slot), m.parameter(0)) },
    parameters: new Float32Array(right ? [1, 0.5, 1, 1] : [1, 1, 1, 1]),
    textures: right ? [undefined, asset] : [asset], samplers: right ? [undefined, sampler] : [sampler] };
}

export async function runTextureResidencyFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;z-index:10000';
  document.body.append(canvas);
  const coarse = texture(2), fine = texture(0), masked = texture(2, 64);
  const scenes: SceneBatch[] = [-0.6, 0.6].map((x, i) => {
    const model = multiply(translation(x, 0, 0), scaling(0.45, 0.45, 1));
    return { name: i ? 'right' : 'left', mesh: quad(), capacity: 1, inputMode: 'committed',
      material: binding(coarse, !!i), deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
      instances: [{ entityId: i + 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
        color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 }] };
  });
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scenes, { textureBudgetBytes: 360, worldQueries: { maxRays: 4 },
      sunShadow: { resolution: 256, focusBounds: { min: [-1.5, -1, -0.5], max: [1.5, 1, 0.5] } } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    active.setMaterialParameters('left', new Float32Array([0.5, 1, 1, 1]));
    const capture = async (name: string) => {
      active.render([0, 0, 4], [0, 0, 0]);
      const [hdr, depth, query] = await Promise.all([
        active.probeHdr([{ x: 46, y: 64 }, { x: 82, y: 64 }]),
        active.probeDepth([[-0.6, 0, 0], [0.6, 0, 0]]),
        active.traceRays([-0.6, 0.6].map(x => ({ origin: [x, 0, 4] as [number, number, number],
          direction: [0, 0, -1] as [number, number, number], tMax: 8 }))),
      ]);
      return { name, hdr, depth, hits: query.hits, png: canvas.toDataURL(),
        revision: active.diagnostics().materialRevision, variants: active.diagnostics().materialVariants,
        images: active.diagnostics().materials, uploads: active.diagnostics().materialTextureUploads };
    };
    const initial = await capture('coarse');
    const cancelled = active.stageTexture(fine);
    const cancelBytes = cancelled.upload(32);
    const beforeCancel = await capture('partial-cancel');
    cancelled.dispose();
    const afterCancel = await capture('cancelled');
    const staged = active.stageTexture(fine), ticks: number[] = [];
    const partial: Awaited<ReturnType<typeof capture>>[] = [];
    let partialRejections = 0;
    while (!staged.complete) {
      ticks.push(staged.upload(32));
      if (!staged.complete) {
        try { active.replaceTexture(coarse, fine); }
        catch (error) {
          if (!String(error).includes('MATERIAL_TEXTURE_PENDING_UPLOAD')) throw error;
          partialRejections++;
        }
        partial.push(await capture('partial-' + ticks.length));
      }
      if (ticks.length > 32) throw new Error('TEXTURE_FIXTURE_NO_UPLOAD_PROGRESS');
    }
    // This material is allocated before its async pipeline reuse completes.
    // A global residency change must also update that pending material handle.
    const pendingMaterial = active.setMaterial('right', binding(coarse, true));
    const revisionBefore = active.diagnostics().materialRevision;
    const replacement = active.replaceTexture(coarse, fine);
    const revisionAfter = active.diagnostics().materialRevision;
    staged.dispose();
    await pendingMaterial;
    const upgraded = await capture('fine');
    const unused = active.replaceTexture(coarse, masked);
    const beforeBudget = active.stats.ownedGpuBytes;
    let budgetError = '';
    try { active.stageTexture(texture(0)); } catch (error) { budgetError = String(error); }
    const budgetUnchanged = beforeBudget === active.stats.ownedGpuBytes;
    const maskStage = active.stageTexture(masked), maskTicks: number[] = [];
    while (!maskStage.complete) maskTicks.push(maskStage.upload(8));
    const maskReplacement = active.replaceTexture(fine, masked); maskStage.dispose();
    const holes = await capture('masked');
    const restore = active.stageTexture(coarse);
    while (!restore.complete) restore.upload(8);
    const restoredReplacement = active.replaceTexture(masked, coarse); restore.dispose();
    const restored = await capture('restored');
    const adapter = active.info.adapterInfo, errors = [...active.errors];
    active.dispose();
    return { initial, cancelBytes, beforeCancel, afterCancel, ticks, partial, partialRejections,
      replacement, revisionBefore, revisionAfter, upgraded, unused, budgetError, budgetUnchanged,
      maskTicks, maskReplacement, holes, restoredReplacement, restored, adapter, errors,
      afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
