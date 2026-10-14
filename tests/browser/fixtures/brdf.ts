import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const normal: Vec3 = [0, 0, 1];
const baseColor: [number, number, number, number] = [0.6, 0.3, 0.15, 1];
const target: Vec3 = [0, 0, 0];

function quad(): MeshData {
  const positions: readonly Vec3[] = [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]];
  const vertices = new ArrayBuffer(4 * 80);
  const data = new DataView(vertices);
  for (let i = 0; i < positions.length; i++) {
    const byte = i * 80;
    for (let axis = 0; axis < 3; axis++) data.setFloat32(byte + axis * 4, positions[i][axis], true);
    data.setFloat32(byte + 12, 1, true);
    data.setFloat32(byte + 24, 1, true);
    data.setUint32(byte + 32, 0, true);
    data.setFloat32(byte + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-2, -2, 0], max: [2, 2, 0] } };
}

function pose(metalness: number, roughness: number): InstanceState {
  return { entityId: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: baseColor, metalness, roughness, historyVersion: 0 };
}

function unit(x: number, y: number): Vec3 {
  return [x, y, Math.sqrt(1 - x * x - y * y)];
}

/** One 1×1 image means the sole sample ray goes through the authored world origin. */
export async function runBrdfFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;';
  document.body.appendChild(canvas);
  const batch: SceneBatch = { name: 'brdf-quad', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [pose(0, 0.5)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch]);
    const sample = async (metalness: number, roughness: number, view: Vec3,
      light: Vec3, sunStrength: number): Promise<number[]> => {
      batch.instances = [pose(metalness, roughness)];
      renderer!.setSun({ direction: light, color: [sunStrength, sunStrength, sunStrength] });
      renderer!.render([view[0] * 6, view[1] * 6, view[2] * 6], target);
      const size = renderer!.stats;
      if (size.width !== 1 || size.height !== 1) {
        throw new Error(`BRDF_CENTER_PIXEL_REQUIRED: actual ${size.width}x${size.height}`);
      }
      return (await renderer!.probeHdr([{ x: Math.floor(size.width / 2), y: Math.floor(size.height / 2) }]))[0];
    };

    const normalIncidence: { metalness: number; roughness: number; ambient: number[]; lit: number[] }[] = [];
    for (const roughness of [0.5, 1]) for (const metalness of [0, 1]) {
      const ambient = await sample(metalness, roughness, normal, normal, 0);
      const lit = await sample(metalness, roughness, normal, normal, 1);
      normalIncidence.push({ metalness, roughness, ambient, lit });
    }
    const smooth = {
      ambient: await sample(0, 0.04, normal, normal, 0),
      lit: await sample(0, 0.04, normal, normal, 1),
    };

    const anglePairs: readonly [Vec3, Vec3][] = [
      [unit(0.4, 0), unit(-0.25, 0)],
      [unit(0.15, 0.35), unit(-0.35, 0.15)],
    ];
    const reciprocal: { view: Vec3; light: Vec3; ambient: number[]; forward: number[]; reverse: number[] }[] = [];
    for (const [view, light] of anglePairs) {
      // The PBR ambient term is independent of view and sun; use a measured zero-sun
      // baseline for each pair before comparing swapped direct-light responses.
      const ambient = await sample(0, 0.5, view, light, 0);
      const forward = await sample(0, 0.5, view, light, 1);
      const reverse = await sample(0, 0.5, light, view, 1);
      reciprocal.push({ view, light, ambient, forward, reverse });
    }
    const frames = renderer.stats.frames, errors = [...renderer.errors];
    renderer.dispose();
    return { baseColor, normalIncidence, smooth, reciprocal, frames, errors,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
