/** Frozen measurement formulas for the joint GI acceptance harness
 * (docs/benchmarks.md). Inputs are linear HDR radiance before exposure; the
 * functions are pure so the same numbers can be recomputed from saved series. */

export type Rgb = readonly [number, number, number] | readonly number[];

/** Rec. 709 relative luminance of linear radiance. */
export function luminance(rgb: Rgb): number {
  if (rgb.length < 3 || ![rgb[0], rgb[1], rgb[2]].every(Number.isFinite)) throw new RangeError('GI_ACCEPTANCE_RGB');
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

export function mean(values: readonly number[]): number {
  if (!values.length || !values.every(Number.isFinite)) throw new RangeError('GI_ACCEPTANCE_VALUES');
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
}

/** `mean(abs(test - reference)) / max(mean(abs(reference)), epsilon)` over all
 * samples and channels. */
export function normalizedMae(test: readonly Rgb[], reference: readonly Rgb[], epsilon: number): number {
  if (test.length !== reference.length || !test.length || !(epsilon > 0)) throw new RangeError('GI_ACCEPTANCE_MAE');
  let error = 0, magnitude = 0, count = 0;
  for (let i = 0; i < test.length; i++) for (let c = 0; c < 3; c++) {
    if (!Number.isFinite(test[i][c]) || !Number.isFinite(reference[i][c])) throw new RangeError('GI_ACCEPTANCE_MAE');
    error += Math.abs(test[i][c] - reference[i][c]); magnitude += Math.abs(reference[i][c]); count++;
  }
  return (error / count) / Math.max(magnitude / count, epsilon);
}

/** Relative steady-state change `|b - a| / max(|a|, epsilon)`. */
export function relativeChange(a: number, b: number, epsilon: number): number {
  if (!Number.isFinite(a) || !Number.isFinite(b) || !(epsilon > 0)) throw new RangeError('GI_ACCEPTANCE_CHANGE');
  return Math.abs(b - a) / Math.max(Math.abs(a), epsilon);
}

/** Fixed-camera fluctuation `(max - min) / mean`; null for a near-dark ROI. */
export function fluctuation(series: readonly number[], darkThreshold: number): number | null {
  const average = mean(series);
  if (average <= darkThreshold) return null;
  return (Math.max(...series) - Math.min(...series)) / average;
}

export interface ResponseTime {
  /** Mean of the frames before the event and of the final settled window. */
  readonly initial: number;
  readonly final: number;
  readonly change: number;
  /** Frames after the event until the response first reaches `fraction` of the
   * final change (frame 1 is the first frame rendered after the event). */
  readonly firstFrames: number | null;
  /** Frames until the response enters and then stays within the band of
   * `1 - fraction` of the change around the final value (conservative T90). */
  readonly settledFrames: number | null;
  /** Largest excursion beyond the final value, relative to the change. */
  readonly overshoot: number;
  /** True when the change is below the noise floor and no time is defined. */
  readonly negligible: boolean;
}

/** T90-style response of a luminance series to an event applied before
 * `series[eventIndex]` was rendered. The last `finalWindow` frames define the
 * final value; the `baselineWindow` frames before the event the initial one. */
export function responseTime(series: readonly number[], eventIndex: number,
  options: { fraction?: number; finalWindow?: number; baselineWindow?: number; minimumChange?: number } = {}): ResponseTime {
  const fraction = options.fraction ?? 0.9;
  const finalWindow = options.finalWindow ?? 8, baselineWindow = options.baselineWindow ?? 4;
  if (!Number.isInteger(eventIndex) || eventIndex < baselineWindow || eventIndex + finalWindow > series.length
    || !(fraction > 0 && fraction < 1)) throw new RangeError('GI_ACCEPTANCE_RESPONSE');
  const initial = mean(series.slice(eventIndex - baselineWindow, eventIndex));
  const final = mean(series.slice(series.length - finalWindow));
  const change = final - initial;
  const minimumChange = options.minimumChange ?? 0;
  if (Math.abs(change) <= minimumChange) {
    return { initial, final, change, firstFrames: null, settledFrames: null, overshoot: 0, negligible: true };
  }
  const band = (1 - fraction) * Math.abs(change);
  let firstFrames: number | null = null, lastOutside = -1, overshoot = 0;
  for (let i = eventIndex; i < series.length; i++) {
    const progress = (series[i] - initial) / change;
    if (firstFrames === null && progress >= fraction) firstFrames = i - eventIndex + 1;
    if (Math.abs(series[i] - final) > band) lastOutside = i;
    overshoot = Math.max(overshoot, progress - 1);
  }
  const settled = lastOutside + 1;
  return { initial, final, change, firstFrames,
    settledFrames: settled < series.length - finalWindow + 1 ? Math.max(1, settled - eventIndex + 1) : null,
    overshoot, negligible: false };
}

/** Fraction of pixels still differing from their final value by more than
 * `fraction` of their own change (or `absolute`, whichever is larger) at one frame.
 * Pixels whose change is below `absolute` count only if they move by more than it. */
export function ghostArea(current: readonly number[], initial: readonly number[], final: readonly number[],
  options: { fraction?: number; absolute: number }): number {
  if (current.length !== initial.length || current.length !== final.length || !current.length
    || !(options.absolute > 0)) throw new RangeError('GI_ACCEPTANCE_GHOST');
  const fraction = options.fraction ?? 0.1;
  let ghosts = 0;
  for (let i = 0; i < current.length; i++) {
    const limit = Math.max(fraction * Math.abs(final[i] - initial[i]), options.absolute);
    if (Math.abs(current[i] - final[i]) > limit) ghosts++;
  }
  return ghosts / current.length;
}

export interface LeakSummary {
  readonly meanLuminance: number;
  readonly maxLuminance: number;
  /** Fraction of samples brighter than `brightThreshold` (bright-spot area). */
  readonly brightArea: number;
}

/** Absolute leak in a region whose reference radiance is (near) zero. */
export function leakSummary(samples: readonly Rgb[], brightThreshold: number): LeakSummary {
  if (!samples.length || !(brightThreshold > 0)) throw new RangeError('GI_ACCEPTANCE_LEAK');
  const values = samples.map(luminance);
  return { meanLuminance: mean(values), maxLuminance: Math.max(...values),
    brightArea: values.filter(value => value > brightThreshold).length / values.length };
}

export type GateVerdict = 'pass' | 'fail' | 'not-measured';
export interface Gate {
  readonly id: string;
  readonly metric: string;
  readonly value: number | null;
  readonly limit: number;
  readonly comparison: '<=' | '>=';
  readonly verdict: GateVerdict;
  readonly note?: string;
}

export function gate(id: string, metric: string, value: number | null | undefined, comparison: '<=' | '>=',
  limit: number, note?: string): Gate {
  const verdict: GateVerdict = value === null || value === undefined || !Number.isFinite(value) ? 'not-measured'
    : (comparison === '<=' ? value <= limit : value >= limit) ? 'pass' : 'fail';
  return { id, metric, value: value ?? null, limit, comparison, verdict, ...(note ? { note } : {}) };
}
