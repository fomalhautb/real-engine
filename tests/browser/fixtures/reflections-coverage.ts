import type { SceneBatch } from '../../../src/contracts';
import { createCube, type MeshData } from '../../../src/geometry';
import { multiply, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import { instance } from './reflections-temporal';

export const COVERAGE_SIZE = 64;
export const COVERAGE_SPACINGS = [1, 2, 4] as const;
/** Rod half-widths in meters; about 0.45, 0.9 and 1.9 pixels wide at the rods' depth. */
export const ROD_HALF_WIDTHS = [0.012, 0.025, 0.05] as const;
const EYE: [number, number, number] = [0, 0, 3.2];
const TARGET: [number, number, number] = [0.2, 0, 0];
const SPHERE_ID = 1, ROD_IDS = [2, 3, 4] as const;

/** Smooth UV sphere of radius one in the engine's 80-byte source vertex layout. */
function sphere(segments: number, rings: number): MeshData {
  const count = (segments + 1) * (rings + 1);
  const vertices = new ArrayBuffer(count * 80), view = new DataView(vertices), indices: number[] = [];
  for (let ring = 0; ring <= rings; ring++) for (let segment = 0; segment <= segments; segment++) {
    const theta = Math.PI * ring / rings, phi = 2 * Math.PI * segment / segments;
    const point = [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
    const offset = (ring * (segments + 1) + segment) * 80;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(offset + axis * 4, point[axis], true);
      view.setFloat32(offset + 16 + axis * 4, point[axis], true);
    }
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 48, 1, true);
  }
  for (let ring = 0; ring < rings; ring++) for (let segment = 0; segment < segments; segment++) {
    const a = ring * (segments + 1) + segment, b = a + segments + 1;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
  }
  return { vertices, vertexCount: count, indices: new Uint32Array(indices), bounds: { min: [-1, -1, -1], max: [1, 1, 1] } };
}

/** Curved and thin mirror receivers under a uniform bright sky, reconstructed at
 * several anchor spacings. Spacing 1 places an anchor on every pixel. */
export async function runReflectionsCoverageFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${COVERAGE_SIZE}px;height:${COVERAGE_SIZE}px`;
  document.body.appendChild(canvas);
  const mirror = { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) };
  const pixels = Array.from({ length: COVERAGE_SIZE * COVERAGE_SIZE }, (_, i) => ({ x: i % COVERAGE_SIZE, y: Math.floor(i / COVERAGE_SIZE) }));
  const chunks = <T>(values: readonly T[]) => Array.from({ length: Math.ceil(values.length / 1024) }, (_, i) => values.slice(i * 1024, (i + 1) * 1024));
  const runs = [];
  let mask: number[] = [];
  try {
    for (const spacing of COVERAGE_SPACINGS) {
      const ball: SceneBatch = { name: 'curved-receiver', mesh: sphere(64, 32), capacity: 1, inputMode: 'committed',
        instances: [instance(SPHERE_ID, multiply(translation(-0.75, 0, 0), scaling(0.7, 0.7, 0.7)), 0.045)],
        deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: mirror } };
      const rods: SceneBatch = { name: 'thin-receivers', mesh: createCube(), capacity: ROD_IDS.length, inputMode: 'committed',
        instances: ROD_IDS.map((id, index) => instance(id, multiply(translation(0.45 + 0.4 * index, 0, 0),
          scaling(ROD_HALF_WIDTHS[index] * 2, 2, ROD_HALF_WIDTHS[index] * 2)), 0.045)),
        deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: mirror } };
      const columns = Math.ceil(COVERAGE_SIZE / spacing);
      const renderer = await Renderer.create(canvas, [ball, rods], { antiAliasing: 'none',
        worldQueries: { maxRays: COVERAGE_SIZE * COVERAGE_SIZE * 4 },
        reflections: { pixelSpacing: spacing, maxAnchors: columns * columns, samplesPerAnchor: 4, maxDistance: 20,
          maxBytes: 32 * 1024 * 1024, skyRadiance: [1, 1, 1] } });
      try {
        renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
        renderer.render(EYE, TARGET);
        renderer.render(EYE, TARGET);
        if (spacing === 1) {
          mask = [];
          for (const part of chunks(pixels.map((_, i) => i))) {
            const probe = await renderer.probeReflections(part);
            mask.push(...probe.anchors.map(anchor => anchor.identity[0] ? anchor.identity[1] : 0));
          }
        }
        const hdr: number[][] = [];
        for (const part of chunks(pixels)) hdr.push(...await renderer.probeSurfaceHdr(part));
        runs.push({ spacing, hdr, errors: [...renderer.errors], grid: renderer.diagnostics().reflections?.grid });
      } finally {
        renderer.dispose();
        runs.at(-1) && Object.assign(runs.at(-1)!, { ownedBytesAfterDispose: renderer.stats.ownedGpuBytes });
      }
    }
    return { size: COVERAGE_SIZE, ids: { sphere: SPHERE_ID, rods: [...ROD_IDS] }, rodHalfWidths: [...ROD_HALF_WIDTHS], mask,
      runs: runs as (typeof runs[number] & { ownedBytesAfterDispose: number })[] };
  } finally { canvas.remove(); }
}
