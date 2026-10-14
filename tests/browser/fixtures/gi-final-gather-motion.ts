import { createGiDemoScene, GI_PROBE_GRID } from '../../../src/gi-demo-scene';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const width = 960, height = 540;
const eye: Vec3 = [1.523339179904896, 2.963601128661066, 8.021397057187407];
const target: Vec3 = [0, 1.15, -0.35];
const sunDirection: Vec3 = [Math.sin(72 * Math.PI / 180) * Math.cos(52 * Math.PI / 180),
  Math.sin(52 * Math.PI / 180), Math.cos(72 * Math.PI / 180) * Math.cos(52 * Math.PI / 180)];
const linear = (byte: number) => {
  const value = byte / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
};
const sunColor: Vec3 = [linear(255) * 3.2, linear(241) * 3.2, linear(220) * 3.2];
const points: Vec3[] = Array.from({ length: 144 }, (_, index) => {
  const row = Math.floor(index / 12), column = index % 12;
  return [3.2, 0.6 + row * 2.6 / 11, -2.8 + column * 5.6 / 11];
});

function unit(vector: readonly number[]): Vec3 {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}
function cross(a: readonly number[], b: readonly number[]): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function sample(world: Vec3, cameraEye: Vec3, cameraTarget: Vec3) {
  const vp = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150),
    lookAt(cameraEye, cameraTarget, [0, 1, 0]));
  const ndc = transformPoint(vp, world);
  const pixel = { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
    y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
  if (!ndc.every(Number.isFinite) || ndc[2] < 0 || ndc[2] > 1 ||
    pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) return null;
  const forward = unit(cameraTarget.map((value, channel) => value - cameraEye[channel]));
  const right = unit(cross(forward, [0, 1, 0]));
  const up = cross(right, forward), focal = Math.tan(Math.PI / 6.4);
  const x = (2 * (pixel.x + 0.5) / width - 1) * width / height * focal;
  const y = (1 - 2 * (pixel.y + 0.5) / height) * focal;
  const direction = unit(forward.map((value, channel) => value + right[channel] * x + up[channel] * y));
  return { pixel, ray: { origin: cameraEye, direction, tMin: 1e-5, tMax: 150 } satisfies WorldRay };
}

interface MotionResult {
  mode: 'carried' | 'reset';
  motionFrames: number;
  validPairs: number;
  validSamples: number;
  changeRms: number;
  meanRgb: Vec3;
  sampleHash: number;
  ownedGpuBytesFirst: number;
  ownedGpuBytesLast: number;
  errors: string[];
  afterDispose: number;
  png: string;
}

async function run(mode: MotionResult['mode']): Promise<MotionResult> {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px;`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, createGiDemoScene().batches, {
      diffuseGi: { ...GI_PROBE_GRID, maxBytes: 64 * 1024 * 1024,
        finalGather: { pixelSpacing: 4, maxAnchors: 32768, raysPerAnchor: 2,
          maxHistory: 64, maxBytes: 64 * 1024 * 1024 } },
      worldQueries: { maxRays: 65536 }, antiAliasing: 'fxaa',
      sunShadow: { resolution: 2048,
        focusBounds: { min: [-3.5, -0.25, -4], max: [3.5, 4, 4] },
        maxBytes: 16 * 1024 * 1024 },
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    for (let frame = 0; frame < 64; frame++) renderer.render(eye, target);
    const ownedGpuBytesFirst = renderer.stats.ownedGpuBytes;
    let previous = new Map<number, Vec3>();
    let sumSquared = 0, validPairs = 0, validSamples = 0, sampleHash = 2166136261 >>> 0;
    const totalRgb: Vec3 = [0, 0, 0];
    for (let frame = 0; frame < 48; frame++) {
      const displacement = 0.003 * (frame + 1);
      const cameraEye: Vec3 = [eye[0] + displacement, eye[1], eye[2]];
      const cameraTarget: Vec3 = [target[0] + displacement, target[1], target[2]];
      if (mode === 'reset') renderer.invalidateHistory();
      renderer.render(cameraEye, cameraTarget);
      const projected = points.map(world => sample(world, cameraEye, cameraTarget));
      const candidates = projected.flatMap((value, index) => value ? [{ index, ...value }] : []);
      const hits = (await renderer.traceRays(candidates.map(value => value.ray))).hits;
      const visible = candidates.filter((candidate, index) =>
        hits[index]?.batch === 'blue-wall' && hits[index]?.entityId === 4);
      const hdr = await renderer.probeHdr(visible.map(value => value.pixel));
      const current = new Map<number, Vec3>();
      for (let index = 0; index < points.length; index++) {
        const pixel = projected[index]?.pixel;
        for (const value of [frame, index, pixel?.x ?? -1, pixel?.y ?? -1,
          visible.some(candidate => candidate.index === index) ? 1 : 0]) {
          sampleHash = Math.imul(sampleHash ^ value, 16777619) >>> 0;
        }
      }
      for (let index = 0; index < visible.length; index++) {
        const rgb = hdr[index].slice(0, 3) as Vec3;
        const point = visible[index].index;
        current.set(point, rgb);
        validSamples++;
        for (let channel = 0; channel < 3; channel++) totalRgb[channel] += rgb[channel];
        const prior = previous.get(point);
        if (prior) {
          for (let channel = 0; channel < 3; channel++) sumSquared += (rgb[channel] - prior[channel]) ** 2;
          validPairs++;
        }
      }
      previous = current;
    }
    const png = canvas.toDataURL('image/png');
    const ownedGpuBytesLast = renderer.stats.ownedGpuBytes;
    const errors = [...renderer.errors];
    renderer.dispose();
    return { mode, motionFrames: 48, validPairs, validSamples,
      changeRms: Math.sqrt(sumSquared / Math.max(1, validPairs * 3)),
      meanRgb: totalRgb.map(value => value / Math.max(1, validSamples)) as Vec3,
      sampleHash, ownedGpuBytesFirst, ownedGpuBytesLast, errors,
      afterDispose: renderer.stats.ownedGpuBytes, png };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Sequential lifetimes keep renderer state, GPU ownership and camera sequence independent. */
export async function runFinalGatherMotionFixture() {
  const carried = await run('carried');
  const baseline = await run('reset');
  return { carried, baseline };
}
