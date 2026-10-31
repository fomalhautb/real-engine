/**
 * Independent Node-side references for local light browser specs. Nothing here imports the engine:
 * the formulas restate the documented contract so a wrong engine constant cannot cancel out.
 */
import type { LocalLightDescriptor } from '../../src/local-lights';

export type Vec3 = [number, number, number];

export const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const normalize = (v: readonly number[]): Vec3 => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };

/** saturate(1 − (d/r)⁴)²: reaches zero with zero slope at the range. */
export function rangeWindow(distance: number, range: number): number {
  const ratio = (distance * distance) / (range * range);
  return Math.min(1, Math.max(0, 1 - ratio * ratio)) ** 2;
}

/** E⊥ = colour × intensity × window × cone / max(d², 0.01²) and the unit direction towards the light. */
export function irradiance(light: LocalLightDescriptor, world: readonly number[]): { l: Vec3; e: Vec3; distance: number } {
  const delta = sub(light.position, world), d2 = dot(delta, delta), l = normalize(delta), distance = Math.sqrt(d2);
  if (d2 >= light.range * light.range) return { l, e: [0, 0, 0], distance };
  let factor = rangeWindow(distance, light.range) / Math.max(d2, 1e-4);
  if (light.kind === 'spot') {
    const axis = normalize(light.direction!), cosOuter = Math.cos(light.outerConeAngle!);
    const t = Math.min(1, Math.max(0, (-dot(l, axis) - cosOuter) / (Math.cos(light.innerConeAngle!) - cosOuter)));
    factor *= t * t;
  }
  return { l, e: light.color.map(c => c * light.intensity * factor) as Vec3, distance };
}

/** GGX/Smith-Schlick/Schlick-Fresnel direct response × n·l, the documented engine BRDF. */
export function brdf(base: readonly number[], metallic: number, roughness: number, n: Vec3, v: Vec3, l: Vec3): Vec3 {
  const half = [l[0] + v[0], l[1] + v[1], l[2] + v[2]], hl = Math.max(Math.hypot(half[0], half[1], half[2]), 1e-5);
  const h = half.map(value => value / hl);
  const nv = Math.max(dot(n, v), 0.001), nl = dot(n, l) > 0 ? Math.max(dot(n, l), 0) : 0;
  const nh = Math.max(dot(n, h), 0), vh = Math.max(dot(v, h), 0);
  const a2 = roughness ** 4, denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / Math.max(Math.PI * denominator * denominator, 1e-20);
  const k = (roughness + 1) ** 2 / 8;
  const geometry = (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k));
  return [0, 1, 2].map(c => {
    const f0 = 0.04 + (base[c] - 0.04) * metallic, fresnel = f0 + (1 - f0) * (1 - vh) ** 5;
    const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
    const diffuse = (1 - fresnel) * (1 - metallic) * base[c] / Math.PI;
    return (diffuse + specular) * nl;
  }) as Vec3;
}

/** Henyey-Greenstein phase function for the cosine between the light and view directions. */
export function henyeyGreenstein(cosine: number, g: number): number {
  const denominator = 1 + g * g - 2 * g * cosine;
  return (1 - g * g) / (4 * Math.PI * denominator * Math.sqrt(denominator));
}
