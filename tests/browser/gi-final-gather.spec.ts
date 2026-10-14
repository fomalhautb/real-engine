import { expect, test } from '@playwright/test';
import type { runFinalGatherSkyFixture } from './fixtures/gi-final-gather';
import type { runGiContactReferenceFixture } from './fixtures/gi-contact-reference';
import { writeFile } from 'node:fs/promises';

test('surface final gather preserves sky energy and responds to offscreen occluder removal', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-final-gather.ts') as {
      runFinalGatherSkyFixture: typeof runFinalGatherSkyFixture;
    };
    return module.runFinalGatherSkyFixture();
  });
  console.log('FINAL_GATHER_SKY ' + JSON.stringify({ initial: result.initial.hdr,
    blocked: result.blocked.hdr, restored: result.restoredImmediate.hdr,
    history: result.stable.gi?.finalGather?.historyUpdates,
    errors: result.errors, afterDispose: result.afterDispose }));
  expect(errors).toEqual([]); expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  for (let c = 0; c < 3; c++) {
    // Cosine sampling integrates a constant incident sky exactly: E=pi*L.
    const expected = result.base[c] * 0.96 * result.sky[c];
    for (const sample of [result.initial, result.stable, result.restoredImmediate, result.reenabled]) {
      expect(Math.abs(sample.hdr[c] - expected)).toBeLessThan(0.0002);
    }
    expect(result.blocked.hdr[c]).toBeLessThan(expected * 0.35);
    expect(result.disabled.hdr[c]).toBe(0);
  }
  expect(result.stable.bytes).toBe(result.initial.bytes);
  expect(result.blocked.bytes).toBe(result.initial.bytes);
  expect(result.resized.gi?.finalGather).not.toBeNull();
  expect(result.resized.gi?.finalGather?.enabled).toBe(true);
});

test('surface final gather is compared with matched near-contact path references', async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware',
    'Dense contact accuracy capture requires native hardware; software CI retains sky, emission and base-reference coverage.');
  test.setTimeout(360_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-contact-reference.ts') as {
      runGiContactReferenceFixture: typeof runGiContactReferenceFixture;
    };
    return module.runGiContactReferenceFixture({ finalGather: true });
  });
  const profile = result.samples.flatMap((sample, index) => {
    if (!sample.pixelReference || !result.gather) return [];
    const off = result.cases.off.hdr[index];
    const reference = sample.pixelReference.twelveBounce.mean;
    const previous = result.cases.multi.hdr[index].slice(0, 3).map((x, c) => x - off[c]);
    const gathered = result.gather.multi.hdr[index].slice(0, 3).map((x, c) => x - off[c]);
    return [{ distance: sample.pixelDistanceFromFront, reference, previous, gathered,
      relativeBefore: previous.map((x, c) => Math.abs(x - reference[c]) / reference[c]),
      relativeAfter: gathered.map((x, c) => Math.abs(x - reference[c]) / reference[c]) }];
  });
  console.log('FINAL_GATHER_CONTACT ' + JSON.stringify({ profile,
    diagnostics: result.gather?.multi.gi?.finalGather }));
  const jsonPath = testInfo.outputPath('gi-final-gather-contact.json');
  await writeFile(jsonPath, JSON.stringify({ ...result, profile }, null, 2));
  await testInfo.attach('gi-final-gather-contact.json', { path: jsonPath, contentType: 'application/json' });
  for (const [name, png] of [['before', result.cases.multi.png], ['after', result.gather?.multi.png]] as const) {
    if (!png) continue;
    const path = testInfo.outputPath('gi-contact-' + name + '.png');
    await writeFile(path, Buffer.from(png.split(',')[1], 'base64'));
    await testInfo.attach('gi-contact-' + name + '.png', { path, contentType: 'image/png' });
  }
  expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  expect(result.gather?.multi.gi?.finalGather?.enabled).toBe(true);
  expect(result.gather?.multi.gi?.finalGather?.actualSpacing).toBe(4);
  expect(profile).toHaveLength(5);
  for (const sample of profile) for (const value of sample.gathered) expect(Number.isFinite(value)).toBe(true);
  const meanBefore = profile.flatMap(sample => sample.relativeBefore).reduce((a, b) => a + b, 0) / 15;
  const meanAfter = profile.flatMap(sample => sample.relativeAfter).reduce((a, b) => a + b, 0) / 15;
  expect(meanAfter).toBeLessThan(meanBefore * 0.5);
  expect(meanAfter).toBeLessThan(0.2);
  // Improvement regression, not the complete GI acceptance gate: individual
  // contact channels and moving-camera quality still require further work.
});
