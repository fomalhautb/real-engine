import type { ReflectionsOptions } from '../../src/reflections';

/**
 * Opt-in reflection quality configuration measured against the L05 and D02
 * targets: the edge-aware spatial filter with its defaults (four à-trous levels,
 * footprint 0.3), the lobe-motion response and a 64-frame stationary history.
 * Fixtures merge it over their own temporal min/max (8/32).
 */
export const REFLECTION_QUALITY_OPTIONS: ReflectionsOptions = Object.freeze({
  spatialFilter: {},
  temporal: { lobeMotionResponse: 0.25, stationaryHistoryFrames: 64 },
});
