import type { SceneBatch } from '../../../src/contracts';
import { createCube } from '../../../src/geometry';
import { lookAt, multiply, perspective, rotationX, scaling, transformPoint, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { WorldRay } from '../../../src/world-query';
import { instance, quad } from './reflections-temporal';

export const SCREEN_SIZE = 48;
export const SCREEN_EYE: [number, number, number] = [0, 1.2, 3];
const TARGET: [number, number, number] = [0, 0.2, 0];
const FLOOR_ID = 1, BOX_ID = 2, PANEL_ID = 3;
const BOX_MODEL = multiply(translation(0, 0.35, 0), scaling(0.7, 0.7, 0.7));
/** Unlit panel above the view frustum; its floor reflection is on screen. */
const PANEL_MODEL = multiply(translation(0, 2.6, -1), multiply(rotationX(Math.PI / 2), scaling(1.2, 0.6, 1)));

function cornersOffscreen(model: Float32Array, viewProjection: Float32Array): boolean {
  const corners = [-1, 1].flatMap(x => [-1, 1].flatMap(y => [-1, 1].map(z => transformPoint(model, [x * 0.5, y * 0.5, z * 0.5]))));
  return corners.every(corner => transformPoint(viewProjection, corner)[1] > 1);
}

/** A mirror floor with a lit box standing on it (near-contact, on screen) and an
 * emissive panel above the frustum. World-only and screen-supplemented frames
 * of the same static scene, plus a camera cut. */
export async function runReflectionsScreenFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${SCREEN_SIZE}px;height:${SCREEN_SIZE}px`;
  document.body.appendChild(canvas);
  const floor: SceneBatch = { name: 'screen-mirror-floor', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [instance(FLOOR_ID, multiply(rotationX(-Math.PI / 2), scaling(4, 4, 1)), 0.045)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } } };
  const box: SceneBatch = { name: 'screen-contact-box', mesh: createCube(), capacity: 1, inputMode: 'committed',
    instances: [instance(BOX_ID, BOX_MODEL, 0.6, 0, [0.8, 0.35, 0.2, 1])], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: DEFAULT_MATERIAL_DEFINITION } };
  const panel: SceneBatch = { name: 'screen-offscreen-panel', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [instance(PANEL_ID, PANEL_MODEL, 1, 0, [0.4, 1.6, 2.4, 1])], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.instanceColor } } };
  const viewProjection = multiply(perspective(Math.PI / 3.2, 1, 0.1, 150), lookAt(SCREEN_EYE, TARGET, [0, 1, 0]));
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [floor, box, panel], { antiAliasing: 'none',
      worldQueries: { maxRays: SCREEN_SIZE * SCREEN_SIZE * 4 },
      reflections: { pixelSpacing: 1, maxAnchors: SCREEN_SIZE * SCREEN_SIZE, samplesPerAnchor: 4, maxDistance: 20,
        maxBytes: 16 * 1024 * 1024, skyRadiance: [0.04, 0.05, 0.07], screenTrace: { maxDistance: 4, steps: 16 } } });
    const active = renderer;
    active.setSun({ direction: [0.35, 1, 0.8], color: [2.5, 2.4, 2.2] });
    const all = Array.from({ length: SCREEN_SIZE * SCREEN_SIZE }, (_, i) => i);
    const probeAll = async () => {
      const anchors = [];
      for (let start = 0; start < all.length; start += 1024) anchors.push(...(await active.probeReflections(all.slice(start, start + 1024))).anchors);
      return anchors.map(anchor => ({ valid: anchor.valid, identity: anchor.identity[1], position: anchor.position, normal: anchor.normal,
        reflected: anchor.reflectedIdentity[0] ? anchor.reflectedIdentity[1] : 0, raw: anchor.rawRadiance,
        distance: anchor.meanReflectedDistance, screen: anchor.screen }));
    };
    active.setReflectionsScreenTraceEnabled(false);
    for (let i = 0; i < 3; i++) active.render(SCREEN_EYE, TARGET);
    const worldOnly = { anchors: await probeAll(), state: active.diagnostics().reflections?.screenTrace };
    active.setReflectionsScreenTraceEnabled(true);
    for (let i = 0; i < 2; i++) active.render(SCREEN_EYE, TARGET);
    const supplemented = { anchors: await probeAll(), state: active.diagnostics().reflections?.screenTrace };
    // The rendered surface at each mirror ray's hit, for anchors that reflect the box.
    const mirrorRays: WorldRay[] = [];
    const rayAnchors: number[] = [];
    supplemented.anchors.forEach((anchor, i) => {
      if (!anchor.valid || anchor.reflected !== BOX_ID || mirrorRays.length >= 1024) return;
      const view = anchor.position.map((value, axis) => value - SCREEN_EYE[axis]), length = Math.hypot(...view);
      const w = view.map(value => value / length), n = anchor.normal, dot = w[0] * n[0] + w[1] * n[1] + w[2] * n[2];
      mirrorRays.push({ origin: anchor.position.map((value, axis) => value + n[axis] * 0.0001) as [number, number, number],
        direction: w.map((value, axis) => value - 2 * dot * n[axis]) as [number, number, number], tMax: 20 });
      rayAnchors.push(i);
    });
    const traced = await active.traceRays(mirrorRays);
    const hitPixels = traced.hits.map(hit => {
      if (!hit) return null;
      const ndc = transformPoint(viewProjection, hit.position);
      return { x: Math.min(SCREEN_SIZE - 1, Math.floor((ndc[0] * 0.5 + 0.5) * SCREEN_SIZE)),
        y: Math.min(SCREEN_SIZE - 1, Math.floor((0.5 - ndc[1] * 0.5) * SCREEN_SIZE)), entity: hit.entityId, distance: hit.distance };
    });
    const surfaced = hitPixels.filter(pixel => pixel !== null) as { x: number; y: number; entity: number; distance: number }[];
    const hdr = surfaced.length ? await active.probeSurfaceHdr(surfaced.map(pixel => ({ x: pixel.x, y: pixel.y }))) : [];
    let next = 0;
    const mirrorHits = hitPixels.map((pixel, i) => ({ anchor: rayAnchors[i], pixel, hdr: pixel ? hdr[next++] : null }));
    // An explicit camera cut invalidates the previous-frame copy for one frame.
    active.invalidateHistory();
    active.render(SCREEN_EYE, TARGET);
    const cut = { anchors: await probeAll(), state: active.diagnostics().reflections?.screenTrace };
    const errors = [...active.errors];
    const diagnostics = active.diagnostics().reflections;
    active.dispose();
    return { adapter: active.info.adapterInfo, size: SCREEN_SIZE, eye: SCREEN_EYE, ids: { floor: FLOOR_ID, box: BOX_ID, panel: PANEL_ID },
      panelOffscreen: cornersOffscreen(PANEL_MODEL, viewProjection), worldOnly, supplemented, cut, mirrorHits, diagnostics, errors,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
