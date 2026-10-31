import { describe, expect, it } from 'vitest';
import { fluctuation, gate, ghostArea, leakSummary, luminance, normalizedMae, relativeChange,
  responseTime } from '../src/gi-acceptance';

describe('GI acceptance formulas', () => {
  it('uses Rec. 709 luminance and the frozen normalized MAE', () => {
    expect(luminance([1, 1, 1])).toBeCloseTo(1, 12);
    expect(luminance([0, 1, 0])).toBeCloseTo(0.7152, 12);
    expect(() => luminance([NaN, 0, 0])).toThrow('GI_ACCEPTANCE_RGB');
    const reference = [[1, 2, 3], [3, 2, 1]];
    expect(normalizedMae(reference, reference, 1e-3)).toBe(0);
    // mean |error| = 0.5, mean |reference| = 2.
    expect(normalizedMae([[1.5, 2.5, 3.5], [2.5, 1.5, 0.5]], reference, 1e-3)).toBeCloseTo(0.25, 12);
    // Epsilon bounds the denominator for near-black references.
    expect(normalizedMae([[0.01, 0.01, 0.01]], [[0, 0, 0]], 0.1)).toBeCloseTo(0.1, 12);
    expect(() => normalizedMae([[1, 1, 1]], [], 1)).toThrow('GI_ACCEPTANCE_MAE');
    expect(relativeChange(2, 2.1, 1e-3)).toBeCloseTo(0.05, 12);
  });

  it('measures T90 by first crossing and by settling into the final band', () => {
    // Event before index 4; exponential approach with ratio 0.5 per frame.
    const series = [0, 0, 0, 0, ...Array.from({ length: 20 }, (_, i) => 1 - 0.5 ** (i + 1))];
    const response = responseTime(series, 4, { finalWindow: 4 });
    expect(response.initial).toBe(0);
    expect(response.final).toBeCloseTo(1, 4);
    // 1 - 0.5^4 = 0.9375 is the first value at or above 90%.
    expect(response).toMatchObject({ firstFrames: 4, settledFrames: 4, negligible: false });
    // An overshoot that returns late settles later than it first crosses.
    const ringing = [1, 1, 1, 1, 2.0, 1.4, 1.0, 1.5, 1.95, 2, 2, 2, 2, 2];
    expect(responseTime(ringing, 4, { finalWindow: 4 })).toMatchObject({ firstFrames: 1, settledFrames: 5 });
    // Decreasing changes are symmetric.
    expect(responseTime(series.map(value => 1 - value), 4, { finalWindow: 4 }).firstFrames).toBe(4);
    // Never settles before the final window: not proven.
    const late = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
    expect(responseTime(late, 4, { finalWindow: 4 })).toMatchObject({ firstFrames: 5, settledFrames: 5 });
    const unsettled = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    expect(responseTime(unsettled, 4, { finalWindow: 4 }).settledFrames).toBeNull();
    expect(responseTime([1, 1, 1, 1, 1, 1, 1, 1], 4, { finalWindow: 4, minimumChange: 1e-3 }).negligible).toBe(true);
    expect(() => responseTime(series, 2)).toThrow('GI_ACCEPTANCE_RESPONSE');
  });

  it('reports fluctuation, ghost area, leaks and gate verdicts', () => {
    expect(fluctuation([1, 1.02, 0.99, 1.01], 0.01)).toBeCloseTo(0.03 / 1.005, 12);
    expect(fluctuation([0.001, 0.002], 0.01)).toBeNull();
    // Pixel 0 still holds half its change, pixel 1 has settled, pixel 2 never changed.
    expect(ghostArea([0.5, 0.05, 1], [1, 1, 1], [0, 0, 1], { absolute: 0.01 })).toBeCloseTo(1 / 3, 12);
    // Unchanged pixels count when they move by more than the absolute floor.
    expect(ghostArea([1.5], [1], [1], { absolute: 0.01 })).toBe(1);
    expect(leakSummary([[0, 0, 0], [0.2, 0.2, 0.2]], 0.1)).toEqual({ meanLuminance: 0.1, maxLuminance: 0.2, brightArea: 0.5 });
    expect(gate('L03', 'gi.t90', 12, '<=', 15).verdict).toBe('pass');
    expect(gate('L03', 'gi.t90', 16, '<=', 15).verdict).toBe('fail');
    expect(gate('L03', 'gi.t90', null, '<=', 15).verdict).toBe('not-measured');
  });
});
