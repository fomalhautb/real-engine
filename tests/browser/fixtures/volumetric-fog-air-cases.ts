import type { MediumCoefficients } from '../../../src/participating-media';

/** Plain data shared by the F04 air-lighting fixture and its Node-side spec (no renderer imports). */
export type Vec3 = [number, number, number];
export interface AirRectangle { readonly name: string; readonly corner: Vec3; readonly u: Vec3; readonly v: Vec3 }

export const AIR_SIZE = 64;
export const AIR_EYE: Vec3 = [0, 0, 6];
export const AIR_TARGET: Vec3 = [0, 0, 0];
export const AIR_MAX_VIEW_DEPTH = 8;
/** The receiver is the plane z = 0 in every camera case (planar depth 6). */
export const AIR_RECEIVER_DEPTH = 6;
export const AIR_ROI = [{ x: 32, y: 32 }, { x: 20, y: 24 }, { x: 44, y: 40 }, { x: 12, y: 52 }, { x: 54, y: 9 }];
/** One screen column from top to bottom: rays toward the courtyard opening first. */
export const AIR_COLUMN = [{ x: 32, y: 4 }, { x: 32, y: 16 }, { x: 32, y: 32 }, { x: 32, y: 48 }, { x: 32, y: 60 }];
export const AIR_SKY: Vec3 = [0.6, 0.9, 1.4];
export const AIR_RED: Vec3 = [3, 0.2, 0.1];
export const AIR_MEDIUM: MediumCoefficients = { extinction: 0.25, albedo: [0.7, 0.7, 0.7], emission: [0, 0, 0], anisotropy: 0 };
export const AIR_MEDIUM_FORWARD: MediumCoefficients = { ...AIR_MEDIUM, anisotropy: 0.6 };

/**
 * Open field without geometry: every probe ray misses, so every probe holds the same sky-only SH, and
 * the fog spans the whole grid depth. maxDistance must exceed the probe spacing: misses are recorded at
 * maxDistance in the visibility moments, so a short range would read as occlusion between probes.
 */
export const AIR_OPEN_GRID = { origin: [-4.5, -4.5, -2.5] as Vec3, spacing: [3, 3, 3] as Vec3, counts: [4, 4, 4] as Vec3,
  raysPerProbe: 64, maxDistance: 20, skyRadiance: AIR_SKY };
export const AIR_OPEN_DEPTH = 8;
/** Room x,y in [-5, 5], z in [0, 8]; probes inside, the camera (z = 6) inside the field. */
export const AIR_ROOM_GRID = { origin: [-4, -4, 0.25] as Vec3, spacing: [4, 4, 3] as Vec3, counts: [3, 3, 3] as Vec3,
  raysPerProbe: 128, maxDistance: 20, skyRadiance: AIR_SKY };
/** Planar depth of the field's far boundary (z = 0.25) from the eye: deeper cells fade to the author sky term. */
export const AIR_ROOM_FIELD_DEPTH = 5.75;

// Walls overlap by 0.1 so no probe ray slips through a seam; every wall is black and unlit
// unless stated, so probes see only the sky (through the opening) or the red wall.
const E = 0.1;
export const AIR_ROOM_WALLS: readonly AirRectangle[] = [
  { name: 'floor', corner: [-5 - E, -5, -E], u: [10 + 2 * E, 0, 0], v: [0, 0, 8 + 2 * E] },
  { name: 'ceiling', corner: [-5 - E, 5, -E], u: [10 + 2 * E, 0, 0], v: [0, 0, 8 + 2 * E] },
  { name: 'left', corner: [-5, -5 - E, -E], u: [0, 10 + 2 * E, 0], v: [0, 0, 8 + 2 * E] },
  { name: 'right', corner: [5, -5 - E, -E], u: [0, 10 + 2 * E, 0], v: [0, 0, 8 + 2 * E] },
  { name: 'far', corner: [-5 - E, -5 - E, 0], u: [10 + 2 * E, 0, 0], v: [0, 10 + 2 * E, 0] },
  // Behind the camera and outside the frustum: black, or red in the off-screen colour case.
  { name: 'back', corner: [-5 - E, -5 - E, 8], u: [10 + 2 * E, 0, 0], v: [0, 10 + 2 * E, 0] },
];

/** Mirror case: local fog behind the camera, seen only by reflection rays off the mirror at z = 0. */
export const AIR_MIRROR = {
  size: 32, eye: [0, 0, 3] as Vec3, pixels: [{ x: 16, y: 16 }, { x: 15, y: 16 }, { x: 17, y: 15 }],
  emitter: [2, 1, 0.5] as Vec3, emitterZ: 6, sky: [0.2, 0.5, 1] as Vec3, fogSteps: 128,
  box: { min: [-3, -3, 3.5] as Vec3, max: [3, 3, 4.5] as Vec3 },
  medium: { extinction: 0.5, albedo: [0.8, 0.8, 0.8], emission: [0, 0, 0], anisotropy: 0 } as MediumCoefficients,
  grid: { origin: [-2, -2, 3.25] as Vec3, spacing: [4, 4, 1.5] as Vec3, counts: [2, 2, 2] as Vec3,
    raysPerProbe: 64, maxDistance: 20, skyRadiance: [0.2, 0.5, 1] as Vec3 },
};
