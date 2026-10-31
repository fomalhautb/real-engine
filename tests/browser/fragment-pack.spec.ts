import { expect, test } from '@playwright/test';
import type { runFragmentPackFixture } from './fixtures/fragment-pack';

test('surfacePackSnorm2x16 writes exact WGSL snorm words into every integer fragment output lane', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/benchmark.html');
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'fragment-pack.ts') as { runFragmentPackFixture: typeof runFragmentPackFixture };
    return fixture.runFragmentPackFixture();
  });
  // The builtinReceiver layout is the platform repro: it is recorded, not asserted, because the
  // Apple/Metal target zeroes or scrambles it while conforming adapters match the reference.
  await testInfo.attach('fragment-pack.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]);
  for (const layout of ['receiver', 'allLanes'] as const) {
    expect(result.results[layout].mismatches).toEqual([]);
    expect(result.results[layout].floatMismatches).toBe(0);
  }
});
