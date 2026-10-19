import { Renderer } from '../../../src/renderer';
import { createCube } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import type { SceneBatch } from '../../../src/contracts';
import type { FogVolume } from '../../../src/volumetric-fog';
import { composeMediumSegments, composeWithSurface, integrateUniformMedium, type MediumSegment } from '../../../src/participating-media';

export async function runFogShadowFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:64px;height:64px'; document.body.appendChild(canvas);
  const mesh = createCube(), positions = new Float32Array(mesh.vertexCount * 3);
  const vertexData = new DataView(mesh.vertices);
  for (let v = 0; v < mesh.vertexCount; v++) for (let joint = 0; joint < 4; joint++) vertexData.setUint32(v * 80 + 32 + joint * 4, 0, true);
  for (let v = 0; v < mesh.vertexCount; v++) positions[v * 3] = 2;
  const model = multiply(translation(0, 3, 3), scaling(20, 0.2, 20));
  const batch: SceneBatch = { name: 'offscreen-masked-roof', mesh, capacity: 1, inputMode: 'committed',
    instances: [{ entityId: 1, model, normalMatrix: normalMatrix(model), bones: [identity()], deformScalars: [0],
      color: [1, 1, 1, 1], roughness: 1, metalness: 0, historyVersion: 0 }],
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] },
    morphTargets: [{ positions, normals: new Float32Array(mesh.vertexCount * 3) }],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, alphaMode: 'mask', alphaCutoff: 0.5, baseColor: m.parameter(0) },
      parameters: new Float32Array([0.5, 0.5, 0.5, 0]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], { sunShadow: { resolution: 1024, focusBounds: { min: [-12, -4, -9], max: [12, 5, 15] } },
      worldQueries: { maxRays: 128 }, diffuseGi: { origin: [-1, -1, 1], spacing: [2, 2, 2], counts: [2, 2, 2], raysPerProbe: 16, maxDistance: 20, skyRadiance: [0.1, 0.2, 0.3] },
      volumetricFog: { pixelSize: 1, slices: 128, maxViewDepth: 6, maxFroxels: 64 * 64 * 128 } });
    const active = renderer, eye: [number, number, number] = [0, 0, 6], target: [number, number, number] = [0, 0, 0];
    const pixels = [16, 32, 48].flatMap(y => [16, 32, 48].map(x => ({ x, y })));
    const render = async () => { active.render(eye, target); return active.probeHdr(pixels); };
    active.setSun({ direction: [0, 1, 0.1], color: [0, 0, 0] });
    const vacuum = await render();
    const firstGi = await active.probeDiffuseGi();
    const volumes: FogVolume[] = [
      { id: 1, shape: { kind: 'box', min: [-4, -4, 1], max: [4, 4, 5] }, coefficients: { extinction: 0.2, albedo: [0, 0, 0], emission: [0.12, 0.02, 0.06] } },
      { id: 2, shape: { kind: 'sphere', center: [0, 0, 3], radius: 1.25 }, coefficients: { extinction: 0.35, albedo: [0, 0, 0], emission: [0.03, 0.1, 0.06] } },
    ];
    active.setVolumetricFogMedia({ volumes });
    const local = await render();
    // Independent high-resolution world-space quadrature, not the GPU's cell partition.
    const reference = pixels.map((pixel, index) => {
      const tangent = Math.tan(Math.PI / 6.4);
      const ray = [(2 * (pixel.x + 0.5) / 64 - 1) * tangent, (1 - 2 * (pixel.y + 0.5) / 64) * tangent, -1];
      const steps = 16384, delta = 6 / steps, length = Math.hypot(...ray);
      let sum: MediumSegment = { transmittance: 1, scattering: [0, 0, 0] };
      for (let step = 0; step < steps; step++) {
        const depth = (step + 0.5) * delta, p = [ray[0] * depth, ray[1] * depth, 6 - depth];
        let extinction = 0; const emission: [number, number, number] = [0, 0, 0];
        for (const volume of volumes) {
          const shape = volume.shape;
          const inside = shape.kind === 'box' ? p.every((x, axis) => x >= shape.min[axis] && x <= shape.max[axis])
            : Math.hypot(...p.map((x, axis) => x - shape.center[axis])) <= shape.radius;
          if (inside) { extinction += volume.coefficients.extinction; emission.forEach((_, c) => emission[c] += volume.coefficients.emission[c]); }
        }
        sum = composeMediumSegments(sum, integrateUniformMedium({ extinction, albedo: [0, 0, 0], emission }, delta * length, [0, 0, 0]));
      }
      return composeWithSurface(sum, vacuum[index].slice(0, 3) as [number, number, number]);
    });
    const mediaGi = await active.probeDiffuseGi();
    const heightMedium = { extinction: 0.24, albedo: [0, 0, 0] as const, emission: [0.08, 0.12, 0.03] as const };
    active.setVolumetricFogMedia({ global: { coefficients: heightMedium, baseHeight: -1, heightFalloff: 0.4 } });
    const heightFog = await render();
    const heightReference = pixels.map((pixel, index) => {
      const tangent = Math.tan(Math.PI / 6.4);
      const ray = [(2 * (pixel.x + 0.5) / 64 - 1) * tangent, (1 - 2 * (pixel.y + 0.5) / 64) * tangent, -1];
      let densityIntegral = 0;
      for (let i = 0; i < 16384; i++) densityIntegral += Math.exp(-0.4 * Math.max(1 + ray[1] * (i + 0.5) * 6 / 16384, 0));
      return composeWithSurface(integrateUniformMedium(heightMedium, densityIntegral * 6 / 16384 * Math.hypot(...ray), [0, 0, 0]),
        vacuum[index].slice(0, 3) as [number, number, number]);
    });
    active.setVolumetricFogMedia({ global: { coefficients: { extinction: 0.2, albedo: [0.9, 0.7, 0.5], emission: [0, 0, 0] } } });
    active.setSun({ direction: [0, 1, 0.1], color: [6, 4, 2] });
    const lit = await render();
    const materials = active.diagnostics().materials;
    active.setMaterialParameters(batch.name, new Float32Array([0.5, 0.5, 0.5, 1]));
    const blocked = await render();
    batch.instances[0] = { ...batch.instances[0], deformScalars: [1] };
    const morphed = await render();
    batch.instances[0] = { ...batch.instances[0], deformScalars: [0], bones: [translation(2, 0, 0)] };
    const skinned = await render();
    batch.instances = [];
    const deleted = await render();
    const diagnostics = active.diagnostics(), info = active.info;
    const surface = await active.probeSurfaceHdr(pixels);
    active.dispose();
    return { pixels, vacuum, local, reference, lit, blocked, morphed, skinned, deleted, surface,
      heightFog, heightReference, firstGi, mediaGi, materials, diagnostics, info, errors: active.errors, releasedBytes: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
