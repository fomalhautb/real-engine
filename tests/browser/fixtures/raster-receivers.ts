import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { identity, lookAt, multiply, perspective, transformPoint, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

const eye: [number, number, number] = [0, 0, 4], target: [number, number, number] = [0, 0, 0];
const anchorIndices = [13, 15, 17, 19].flatMap(y => [13, 15, 17, 19].map(x => y * 32 + x));

function curvedGrid(): MeshData {
  const cells = 16, side = cells + 1;
  const vertices = new ArrayBuffer(side * side * 80), data = new DataView(vertices);
  const indices = new Uint32Array(cells * cells * 6), uvs = new Float32Array(side * side * 2);
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const i = y * side + x, px = x / cells * 4 - 2, py = y / cells * 4 - 2;
    const z = 0.25 * Math.sin(px * 2.2) * Math.cos(py * 1.7);
    for (const [offset, value] of [[0, px], [4, py], [8, z], [12, 1], [24, 1], [48, 1]]) data.setFloat32(i * 80 + offset, value, true);
    uvs.set([x / cells, y / cells], i * 2);
  }
  for (let y = 0, offset = 0; y < cells; y++) for (let x = 0; x < cells; x++, offset += 6) {
    const a = y * side + x, b = a + side;
    indices.set([a, a + 1, b + 1, a, b + 1, b], offset);
  }
  return { vertices, indices, vertexCount: side * side, uvs, bounds: { min: [-2, -2, -0.25], max: [2, 2, 0.25] } };
}

export async function runRasterReceiverFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;';
  document.body.append(canvas);
  let renderer: Renderer | undefined;
  try {
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'raster-receiver-curved', mesh: curvedGrid(), coverage: 'opaque' },
      { maxClusterTriangles: 24, maxClusterVertices: 48, groupSize: 4, maxLevels: 8 });
    const pose: InstanceState = { entityId: 0xffff_ff01, entityGeneration: 7, model: identity(), normalMatrix: identity(),
      bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 3 };
    const material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([0.5, 0.5, 0.5, 1]),
      normal: m.constant([0.65, 0.5, 0.95, 1]), metallic: 0, roughness: 1 } };
    const batch: SceneBatch = { name: 'receiver', mesh: hierarchy.source, geometryLod: hierarchy, capacity: 1,
      inputMode: 'committed', instances: [pose], queryIdentity: { assetId: 'raster-receiver', primitiveId: 0, topologyVersion: 5 },
      deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material };
    renderer = await Renderer.create(canvas, [batch], { worldQueries: { maxRays: 2048 },
      diffuseGi: { origin: [-1, -1, 0.5], spacing: [2, 2, 1], counts: [2, 2, 2], raysPerProbe: 16,
        skyRadiance: [0.1, 0.2, 0.3], multiBounce: false,
        finalGather: { receiver: 'raster', pixelSpacing: 4, maxAnchors: 1024, raysPerAnchor: 2, maxHistory: 16 } } });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (frames: number) => {
      for (let i = 0; i < frames; i++) renderer!.render(eye, target);
      const anchors = (await renderer!.probeDiffuseGiFinalGather(anchorIndices)).anchors;
      const cut = await renderer!.probeGeometryLod('receiver');
      const valid = anchors.filter(anchor => anchor.valid);
      const depth = valid.length ? await renderer!.probeDepth(valid.map(anchor => anchor.position)) : [];
      const vp = multiply(perspective(Math.PI / 3.2, 1, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
      const depthError = valid.map((anchor, i) => Math.abs(transformPoint(vp, anchor.position)[2] - depth[i].surface));
      return { anchors, cut, depthError, frame: renderer!.stats.frames };
    };
    renderer.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    const fine = await capture(3);
    renderer.setGeometryLod({ maxErrorPixels: 10_000 });
    const coarse = await capture(1), stable = await capture(3);
    // A masked surface must use full source geometry and share PBR coverage.
    await renderer.setMaterial('receiver', { definition: { ...material.definition, alphaMode: 'mask', baseColor: m.constant([0.5, 0.5, 0.5, 0]) } });
    const masked = await capture(1);
    await renderer.setMaterial('receiver', material);
    batch.instances[0] = { ...pose, entityGeneration: 8, historyVersion: 4, model: translation(0.1, 0, 0) };
    const moved = await capture(1);
    const diagnostics = renderer.diagnostics(), errors = [...renderer.errors];
    renderer.dispose();
    return { fine, coarse, stable, masked, moved, diagnostics, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
