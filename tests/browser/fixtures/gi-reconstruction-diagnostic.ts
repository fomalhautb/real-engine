import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, rotationX, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 0, 3], target: Vec3 = [0, 0, 0];
const sky: Vec3 = [0.1, 0.2, 0.3];
const pixels = [
  { x: 64, y: 64 }, // unobstructed white receiver
  { x: 44, y: 64 }, // transparent Mask layer reveals the white receiver
  { x: 84, y: 64 }, // black Unlit layer covers the white receiver
  { x: 5, y: 64 }, // no raster surface
];

function state(id: number, x: number, z: number, size: number): InstanceState {
  const model = multiply(translation(x, 0, z), multiply(rotationX(Math.PI / 2), scaling(size, 1, size)));
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function batch(name: string, id: number, x: number, z: number, size: number,
  shading: 'pbr' | 'unlit' = 'pbr', masked = false): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, inputMode: 'committed',
    instances: [state(id, x, z, size)],
    queryIdentity: { assetId: 'gi-reconstruction-diagnostic/' + name, primitiveId: id, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading,
      alphaMode: masked ? 'mask' : 'opaque', alphaCutoff: 0.5,
      baseColor: m.constant(masked ? [0, 0, 0, 0] : shading === 'unlit' ? [0, 0, 0, 1] : [1, 1, 1, 1]),
      emissive: m.constant([0, 0, 0, 1]), metallic: 0, roughness: 1 } } };
}

/** Test the exact renderer diagnostic against the rendered white-surface HDR.
 * The masked layer is a fully cut-out front plane; the unlit layer is opaque. */
export async function runGiReconstructionDiagnosticFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [
      batch('white-wall', 1, 0, 0, 2),
      batch('masked-hole', 2, -0.5, 0.03, 0.4, 'pbr', true),
      batch('unlit-cover', 3, 0.5, 0.04, 0.4, 'unlit'),
    ], {
      worldQueries: { maxRays: 8192 },
      diffuseGi: { origin: [-2, -2, -2], spacing: [4, 4, 4], counts: [2, 2, 2],
        skyRadiance: sky, maxDistance: 12, multiBounce: false,
        finalGather: { receiver: 'raster', pixelSpacing: 4, maxAnchors: 1024,
          raysPerAnchor: 4, maxHistory: 16, pathSegments: 1 } },
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    for (let frame = 0; frame < 16; frame++) renderer.render(eye, target);
    const capture = async () => {
      const hdrBefore = await renderer!.probeHdr(pixels);
      const before = { frames: renderer!.stats.frames, ownedGpuBytes: renderer!.stats.ownedGpuBytes };
      const diagnostic = await renderer!.probeDiffuseGiReconstruction(pixels, { coverage: true });
      const coverage = diagnostic.coverage!;
      const sampledBranches = pixels.map(p => coverage.branches[p.y * coverage.width + p.x]);
      const { branches: _branches, ...census } = coverage;
      const after = { frames: renderer!.stats.frames, ownedGpuBytes: renderer!.stats.ownedGpuBytes };
      const hdrAfter = await renderer!.probeHdr(pixels);
      return { hdrBefore, diagnostic: { ...diagnostic, coverage: census }, sampledBranches, hdrAfter, before, after };
    };
    const active = await capture();
    renderer.setDiffuseGiFinalGatherEnabled(false);
    renderer.render(eye, target);
    const disabled = await capture();
    const errors = [...renderer.errors];
    renderer.dispose();
    return { sky, pixels, active, disabled, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
