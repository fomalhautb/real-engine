import type { MediumCoefficients } from '../../../src/participating-media';
import type { FogHeightMedium, FogVolume } from '../../../src/volumetric-fog';

type Vec3 = [number, number, number];

/** Plain data shared by the browser fixture and the Node-side spec (no renderer imports). */
export interface FogAnalyticCase {
  readonly label: string;
  readonly global?: FogHeightMedium;
  readonly volumes?: readonly FogVolume[];
  readonly sun: { direction: Vec3; color: Vec3 };
  /**
   * 'f01': a benchmark F01 scene (zero density, uniform fog, pure absorption, uniform emission) gated on
   * both the history-off and the accumulated sequence; 'quadrature': an extra non-uniform scene with a
   * high-sample reference, gated on the history-off sequence and reported (not gated) after accumulation;
   * 'none': finiteness and clamping only.
   */
  readonly reference: 'f01' | 'quadrature' | 'none';
}

const isotropic: MediumCoefficients = { extinction: 0.2, albedo: [0.5, 0.4, 0.3], emission: [0.04, 0.02, 0.01], anisotropy: 0 };
export const FOG_ANALYTIC_CASES: readonly FogAnalyticCase[] = [
  // Sun parallel to the receiver: the whole column is lit.
  { label: 'uniform-isotropic', global: { coefficients: isotropic }, sun: { direction: [0, 1, 0], color: [10, 6, 3] }, reference: 'f01' },
  // Sun behind the camera: view rays meet backward scattering (cos = view . sun < 0).
  { label: 'uniform-forward-g', global: { coefficients: { ...isotropic, emission: [0, 0, 0], anisotropy: 0.6 } },
    sun: { direction: [0, 0.6, 0.8], color: [8, 8, 8] }, reference: 'f01' },
  { label: 'uniform-backward-g', global: { coefficients: { ...isotropic, emission: [0, 0, 0], anisotropy: -0.6 } },
    sun: { direction: [0, 0.6, 0.8], color: [8, 8, 8] }, reference: 'f01' },
  { label: 'height-fog', global: { coefficients: { extinction: 0.3, albedo: [0.6, 0.6, 0.6], emission: [0, 0, 0], anisotropy: 0.3 },
    baseHeight: -1, heightFalloff: 0.5 }, sun: { direction: [0, 0.6, 0.8], color: [6, 4, 2] }, reference: 'quadrature' },
  { label: 'pure-absorption', global: { coefficients: { extinction: 0.5, albedo: [0, 0, 0], emission: [0, 0, 0] } },
    sun: { direction: [0, 0.6, 0.8], color: [8, 8, 8] }, reference: 'f01' },
  { label: 'uniform-emission', global: { coefficients: { extinction: 0.25, albedo: [0, 0, 0], emission: [0.3, 0.2, 0.1] } },
    sun: { direction: [0, 0.6, 0.8], color: [8, 8, 8] }, reference: 'f01' },
  { label: 'zero-density', global: { coefficients: { extinction: 0, albedo: [1, 1, 1], emission: [0, 0, 0] } },
    volumes: [{ id: 7, shape: { kind: 'box', min: [-2, -2, 1], max: [2, 2, 4] }, coefficients: { extinction: 0, albedo: [1, 1, 1], emission: [0, 0, 0] } }],
    sun: { direction: [0, 0.6, 0.8], color: [8, 8, 8] }, reference: 'f01' },
  { label: 'dense-bright', global: { coefficients: { extinction: 40, albedo: [1, 1, 1], emission: [10, 10, 10], anisotropy: 0.95 } },
    sun: { direction: [0, 0.6, 0.8], color: [10000, 10000, 10000] }, reference: 'none' },
  // Sun exactly behind the camera: the sphere's g = -0.95 phase peak meets the view ray
  // and its radiance exceeds binary16, so the output must clamp rather than overflow.
  { label: 'thin-and-peaked', global: { coefficients: { extinction: 1e-7, albedo: [1, 1, 1], emission: [0, 0, 0], anisotropy: 0.95 } },
    volumes: [{ id: 8, shape: { kind: 'sphere', center: [0, 0, 3], radius: 1 }, coefficients: { extinction: 3, albedo: [0.9, 0.9, 0.9], emission: [0, 0, 0], anisotropy: -0.95 } }],
    sun: { direction: [0, 0, 1], color: [1e6, 1e6, 1e6] }, reference: 'none' },
];

