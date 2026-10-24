import { expect, test } from '@playwright/test';
import type { runProgressiveSequenceFixture } from './fixtures/gi-progressive-sequence';
type Fixture = typeof runProgressiveSequenceFixture;

const hash = (x: number) => { x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); return (x ^ (x >>> 16)) >>> 0; };
const random = (x: number) => (hash(x) >>> 8) / 16777216;
function radicalInverse(index: number, base: number) {
  const digits: number[] = []; do { digits.push(index % base); index = Math.floor(index / base); } while (index);
  return digits.reduceRight((v, digit) => (v + digit) / base, 0);
}

test('progressive supplemental directions cross the history cap, consume all rays, and wrap with a new shift', async ({ page }, testInfo) => {
  const pageErrors: string[] = []; page.on('pageerror', e => pageErrors.push(e.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const { runProgressiveSequenceFixture } = await import('/tests/browser/fixtures/' + 'gi-progressive-sequence.ts') as {
      runProgressiveSequenceFixture: Fixture };
    return runProgressiveSequenceFixture();
  });
  await testInfo.attach('progressive-sequence.json', { body: JSON.stringify(result), contentType: 'application/json' });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(result.rows).toHaveLength(60);
  const bases = [2, 3, 5, 7, 11, 13, 17, 19];
  for (const row of result.rows) {
    const counter = row.count === 1_048_576 ? 0 : row.count;
    const rehashed = hash(0x123456 ^ 0x9e3779b9) & 0xffffff;
    const key = row.count === 1_048_576 ? (rehashed === 0x123456 ? 0x123457 : rehashed) : 0x123456;
    expect(row.sequence).toEqual([counter + 1, key]); expect(row.historyCount).toBe(16);
    const seed = hash(key ^ Math.imul(row.segment, 0x85ebca6b));
    for (let i = 0; i < row.raysPerAnchor; i++) {
      const index = counter * row.raysPerAnchor + i;
      const u = (radicalInverse(index, bases[row.segment * 2]) + random(seed ^ 0x68bc21eb)) % 1;
      const v = (radicalInverse(index, bases[row.segment * 2 + 1]) + random(seed ^ 0x02e5be93)) % 1;
      const radius = Math.sqrt(u), angle = 2 * Math.PI * v;
      const expected = [radius * Math.sin(angle), Math.sqrt(1 - u), radius * Math.cos(angle)];
      // Verify the sequence before trigonometry separately. WGSL transcendental
      // accuracy varies by adapter (sin/cos may have 2^-11 absolute error on
      // [-pi, pi], followed here by normalization). This direction threshold is a
      // supported-device regression budget, not a universal WGSL error bound.
      for (let c = 0; c < 2; c++) {
        expect(row.sampling[i][c]).toBeGreaterThanOrEqual(0); expect(row.sampling[i][c]).toBeLessThan(1);
        const error = Math.abs(row.sampling[i][c] - [u, v][c]);
        expect(Math.min(error, 1 - error)).toBeLessThan(0.0000005);
      }
      expect(Math.abs(Math.hypot(...row.directions[i]) - 1)).toBeLessThan(0.000002);
      expect(Math.abs(row.sampling[i][2] - Math.sqrt(row.sampling[i][0]))).toBeLessThan(0.000001);
      expect(Math.abs(row.sampling[i][3] - Math.sqrt(1 - row.sampling[i][0]))).toBeLessThan(0.000001);
      for (let c = 0; c < 3; c++) expect(Math.abs(row.directions[i][c] - expected[c])).toBeLessThan(0.0015);
      expect(row.rayIntervals[i][0]).toBeCloseTo(0.00001, 9); expect(row.rayIntervals[i][1]).toBe(80);
    }
  }
  expect(result.recovery).toHaveLength(4);
  const [failedA, failedB, recovered, continued] = result.recovery;
  for (const failed of [failedA, failedB]) {
    expect(failed.state).toEqual([0, failed.frame, 1, 0]); expect(failed.sequence).toEqual([0, 0]);
  }
  // These fixed frames have distinct keys; 24-bit shifts cannot promise that
  // every possible pair of frames avoids a collision or a query-budget failure.
  expect(failedA.generatedKey).not.toBe(failedB.generatedKey); expect(failedA.uv).not.toEqual(failedB.uv);
  expect(recovered.generatedKey).not.toBe(failedB.generatedKey);
  expect(recovered.sequence).toEqual([1, recovered.generatedKey]); expect(recovered.irradiance).toEqual([4, 4, 4, 1]);
  expect(continued.generatedKey).toBe(recovered.generatedKey); expect(continued.sequence).toEqual([2, recovered.generatedKey]);
  expect(continued.irradiance).toEqual([4, 4, 4, 2]); expect(continued.uv).not.toEqual(recovered.uv);
  expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(result.validationError).toBe(''); expect(result.ownedBytes).toBe(0);
});
