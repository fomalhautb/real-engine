import { expect, test } from '@playwright/test';
import type { runSparseUploadFixture } from './fixtures/sparse-upload';

type Result = Awaited<ReturnType<typeof runSparseUploadFixture>>;
type Capture = Result['first'];
const PER_SLOT = 224 + 128 + 8;
const PREFIX = 'sparse-upload ';

function ranges(capture: Capture, stream: 'instances' | 'palettes' | 'scalars') {
  return capture.writes.filter(write => write.label === PREFIX + stream)
    .map(write => [write.offset, write.bytes]);
}
function params(capture: Capture) {
  return capture.writes.filter(write => write.label === PREFIX + 'params')
    .map(write => [write.offset, write.bytes]);
}
function checked(capture: Capture): void {
  expect(capture.sampleCount).toBe(capture.instances);
  expect(capture.currentError).toBeLessThan(0.0002);
  expect(capture.previousError).toBeLessThan(0.0002);
  expect(capture.referenceError).toBeLessThan(0.0002);
  expect(capture.boundsViolations).toBe(0);
}

test('committed slots upload sparse ranges and preserve independent current/previous poses', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('SPARSE_UPLOAD_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'sparse-upload.ts') as {
      runSparseUploadFixture: typeof runSparseUploadFixture;
    };
    return module.runSparseUploadFixture();
  });
  console.log('SPARSE_UPLOAD_GPU_DIAGNOSTIC ' + JSON.stringify({
    bytes: Object.fromEntries(Object.entries(result).filter(([, value]) => value && typeof value === 'object'
      && 'uploadedBytes' in value).map(([key, value]) => [key, (value as Capture).uploadedBytes])),
    frames: result.frames, errors: result.errors,
  }));

  for (const capture of [result.first, result.still, result.one, result.oneSettle, result.adjacent,
    result.separated, result.fragmented, result.fragmentedSettle, result.reordered, result.deleted,
    result.newGeneration, result.cut, result.resized, result.empty, result.readded, result.liveGeneration, result.mutable,
    result.backToCommitted, result.committedCached, result.afterDuplicate]) checked(capture);
  expect(result.first.uploadedBytes).toBe(256 + 16 + 40 * PER_SLOT);
  expect(result.still.uploadedBytes).toBe(256);
  expect(result.one.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.oneSettle.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.oneCached.uploadedBytes).toBe(256);
  expect(result.one.historyMotion).toBeGreaterThan(0.1);
  expect(result.mutable.historyMotion).toBeGreaterThan(0.1);
  expect(result.one.deformationDispatches).toBe(1);
  expect(result.still.deformationDispatches).toBe(0);
  expect(result.still.writes).toEqual([]);
  expect(result.oneCached.writes).toEqual([]);
  expect(params(result.first)).toEqual([[0, 16]]);
  expect(params(result.one)).toEqual([]);
  expect(params(result.deleted)).toEqual([[0, 16]]);
  expect(params(result.empty)).toEqual([[0, 16]]);
  expect(params(result.readded)).toEqual([[0, 16]]);
  for (const stream of ['instances', 'palettes', 'scalars'] as const) {
    const stride = result.stride[stream];
    expect(ranges(result.one, stream)).toEqual([[20 * stride, stride]]);
    expect(ranges(result.adjacent, stream)).toEqual([[10 * stride, 2 * stride]]);
    expect(ranges(result.separated, stream)).toEqual([[5 * stride, stride], [30 * stride, stride]]);
    expect(ranges(result.fragmented, stream)).toEqual([[0, 39 * stride]]);
  }
  expect(result.adjacent.uploadedBytes).toBe(256 + 2 * PER_SLOT);
  expect(result.separated.uploadedBytes).toBe(256 + 2 * PER_SLOT);
  expect(result.fragmented.uploadedBytes).toBe(256 + 39 * PER_SLOT);
  expect(result.fragmentedSettle.uploadedBytes).toBe(256 + 39 * PER_SLOT);
  expect(result.reordered.uploadedBytes).toBe(256 + 2 * PER_SLOT);
  expect(result.deleted.uploadedBytes).toBe(256 + 16 + 19 * PER_SLOT);
  expect(result.newGeneration.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.cut.uploadedBytes).toBe(256 + 39 * PER_SLOT);
  expect(result.resized.uploadedBytes).toBe(256 + 39 * PER_SLOT);
  expect(result.resized.width).toBeGreaterThan(result.cut.width);
  expect(result.resized.height).toBe(result.cut.height);
  expect(result.empty.uploadedBytes).toBe(256 + 16);
  expect(result.readded.uploadedBytes).toBe(256 + 16 + PER_SLOT);
  expect(result.liveGeneration.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.liveGeneration.historyMotion).toBe(0);
  expect(result.mutable.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.backToCommitted.uploadedBytes).toBe(256 + PER_SLOT);
  expect(result.committedCached.uploadedBytes).toBe(256);
  expect(result.duplicateError).toContain('DUPLICATE_ENTITY_ID');
  expect(result.frameAfterDuplicate).toBe(result.frameBeforeDuplicate);
  expect(result.afterDuplicate.uploadedBytes).toBe(256);
  expect(result.frames).toBe(30);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
