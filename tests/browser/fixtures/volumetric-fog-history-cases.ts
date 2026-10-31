import type { MediumCoefficients } from '../../../src/participating-media';

/** Plain data shared by the F02/F03 history fixture and its Node-side spec (no renderer imports). */
export type Vec3 = [number, number, number];
export interface HistoryQuad { readonly name: string; readonly min: [number, number]; readonly max: [number, number] }

export const HISTORY_SIZE = 64;
export const HISTORY_EYE: Vec3 = [0, 0, 6];
export const HISTORY_TELEPORT_EYE: Vec3 = [1.5, 0.5, 5];
export const HISTORY_TARGET: Vec3 = [0, 0, 0];
export const HISTORY_MAX_FRAMES = 16;
export const HISTORY_MEDIUM: MediumCoefficients = { extinction: 0.2, albedo: [0.8, 0.8, 0.8], emission: [0, 0, 0], anisotropy: 0 };
/** Overhead sun: light travels straight down through the window. */
export const HISTORY_SUN = { direction: [0, 1, 0] as Vec3, color: [6, 6, 6] as Vec3 };
/** Zero-thickness roof at y = 5 (x/z footprints) around a window; the door panel closes the window. */
export const HISTORY_ROOF_Y = 5;
export const HISTORY_WINDOW: HistoryQuad = { name: 'door', min: [0.2, 1], max: [1.4, 4] };
export const HISTORY_ROOF: readonly HistoryQuad[] = [
  { name: 'roof-left', min: [-8, -4], max: [0.2, 10] },
  { name: 'roof-right', min: [1.4, -4], max: [8, 10] },
  { name: 'roof-front', min: [0.2, 4], max: [1.4, 10] },
  { name: 'roof-back', min: [0.2, -4], max: [1.4, 1] },
];
/** The door slides onto the right roof piece, where it changes no lighting. */
export const HISTORY_DOOR_OPEN_OFFSET = 2;
/** Rays through the shaft interior (planar depth 2..5) and through the shadowed side. */
export const HISTORY_SHAFT_PIXEL = { x: 43, y: 32 };
export const HISTORY_SHADOW_PIXEL = { x: 12, y: 40 };
export const HISTORY_PIXELS = [HISTORY_SHAFT_PIXEL, { x: 44, y: 24 }, { x: 42, y: 44 }, HISTORY_SHADOW_PIXEL, { x: 20, y: 20 }];
export const HISTORY_LATTICE = Array.from({ length: 64 }, (_, i) => ({ x: 4 + 8 * (i % 8), y: 4 + 8 * Math.floor(i / 8) }));
/** Froxels (pixelSize 2, 64 slices to 8 m): under the window, under the character, and far from both. */
export const HISTORY_CELLS = { shaft: { x: 21, y: 16, z: 44 }, under: { x: 22, y: 24, z: 42 }, far: { x: 2, y: 4, z: 48 } };
/** Its ray crosses the shaft below the character's shadow column (planar depth 3.2..3.8, y < -0.3). */
export const HISTORY_GHOST_PIXEL = { x: 45, y: 45 };
/** Skinned character (unit cube scaled 0.6) crossing the shaft at planar depth 3.5. */
export const HISTORY_CHARACTER = { scale: 0.6, y: 0, z: 2.5, path: [-0.4, 0.8, 2.6] };
/** Emissive local fog for media add/move/remove, on the shadowed side (rays never meet the shaft). */
export const HISTORY_BOX = { min: [-1.4, -0.4, 2.5] as Vec3, max: [-0.6, 0.4, 3.5] as Vec3, moved: [0, 0.9, 0] as Vec3,
  coefficients: { extinction: 0.3, albedo: [0, 0, 0], emission: [0.5, 0.25, 0.1] } as MediumCoefficients };
export const HISTORY_BOX_PIXEL = { x: 12, y: 32 };
export const HISTORY_MOVED_BOX_PIXEL = { x: 12, y: 14 };
export const HISTORY_STABILITY_FRAMES = 120;
