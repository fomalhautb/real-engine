import { expect, test, type Page } from '@playwright/test';
import type { createMaterialFixture } from './fixtures/materials';

type Fixture = Awaited<ReturnType<typeof createMaterialFixture>>;

async function openFixture(page: Page) {
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  return page.evaluateHandle(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    // Keep this dynamic so Playwright's Node transform never imports WGSL.
    const module = await import('/tests/browser/fixtures/' + 'materials.ts') as { createMaterialFixture: typeof createMaterialFixture };
    return module.createMaterialFixture();
  });
}

function expectHit(sample: { surface: number; shadow: number }) {
  expect(sample.surface).toBeGreaterThan(0);
  expect(sample.surface).toBeLessThan(1);
  expect(sample.shadow).toBeGreaterThan(0);
  expect(sample.shadow).toBeLessThan(1);
}

function expectHole(sample: { surface: number; shadow: number }) {
  expect(sample.surface).toBe(1);
  expect(sample.shadow).toBe(1);
}

async function centerPixel(page: Page): Promise<[number, number, number]> {
  const png = await page.locator('#material-fixture').screenshot();
  return page.evaluate(async base64 => {
    const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    try {
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext('2d', { willReadFrequently: true })!;
      context.drawImage(bitmap, 0, 0);
      const pixel = context.getImageData(Math.floor(bitmap.width / 2), Math.floor(bitmap.height / 2), 1, 1).data;
      return [pixel[0], pixel[1], pixel[2]] as [number, number, number];
    } finally { bitmap.close(); }
  }, png.toString('base64'));
}

test('material coverage is shared by surface and shadow passes, and swaps release resources', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const fixture = await openFixture(page);
  try {
    const opaque = await fixture.evaluate((f: Fixture) => f.textured('opaque'));
    expectHit(opaque.depth[0]);
    expectHit(opaque.depth[1]);
    expect(opaque.diagnostics.materials).toEqual({ materials: 1, images: 1, imageBytes: 20 });

    const masked = await fixture.evaluate((f: Fixture) => f.textured('mask'));
    expectHit(masked.depth[0]);
    expectHole(masked.depth[1]);
    expect(masked.diagnostics.materials).toEqual({ materials: 1, images: 1, imageBytes: 20 });
    expect(masked.diagnostics.materialRevision).toBeGreaterThan(opaque.diagnostics.materialRevision);

    const invalid = await fixture.evaluate((f: Fixture) => f.rejectedTextureSwap());
    expect(invalid.rejection).toMatch(/TEXTURE|slot/i);
    expectHit(invalid.after.depth[0]);
    expectHole(invalid.after.depth[1]);
    expect(invalid.after.diagnostics.materials).toEqual(masked.diagnostics.materials);
    expect(invalid.after.ownedGpuBytes).toBe(masked.ownedGpuBytes);

    const overBudget = await fixture.evaluate((f: Fixture) => f.rejectedBudgetSwap());
    expect(overBudget.rejection).toMatch(/MATERIAL_TEXTURE_BUDGET/);
    expectHit(overBudget.after.depth[0]);
    expectHole(overBudget.after.depth[1]);
    expect(overBudget.after.diagnostics.materials).toEqual(masked.diagnostics.materials);
    expect(overBudget.after.ownedGpuBytes).toBe(masked.ownedGpuBytes);

    const variants = masked.diagnostics.materialVariants;
    for (const red of [0.2, 0.8, 0.4]) {
      const updated = await fixture.evaluate((f: Fixture, value) => f.tint(value), red);
      expect(updated.diagnostics.materialVariants).toBe(variants);
      expect(updated.diagnostics.materials).toEqual(masked.diagnostics.materials);
      expect(updated.ownedGpuBytes).toBe(masked.ownedGpuBytes);
    }

    const plain = await fixture.evaluate((f: Fixture) => f.plain('unlit', false, 'front', [2, 2, 2]));
    expect(plain.diagnostics.materials).toEqual({ materials: 1, images: 0, imageBytes: 0 });
    const stableBytes = plain.ownedGpuBytes;
    for (let i = 0; i < 3; i++) {
      const textured = await fixture.evaluate((f: Fixture) => f.textured('mask'));
      expect(textured.diagnostics.materials).toEqual({ materials: 1, images: 1, imageBytes: 20 });
      const released = await fixture.evaluate((f: Fixture) => f.plain('unlit', false, 'front', [2, 2, 2]));
      expect(released.diagnostics.materials).toEqual({ materials: 1, images: 0, imageBytes: 0 });
      expect(released.diagnostics.materialVariants).toBeLessThanOrEqual(4);
      expect(released.ownedGpuBytes).toBe(stableBytes);
    }
    const sweep = await fixture.evaluate((f: Fixture) => f.sweepVariants());
    for (const result of sweep) {
      expect(result.diagnostics.materialVariants).toBeLessThanOrEqual(4);
      expect(result.ownedGpuBytes).toBe(stableBytes);
      expectHit(result.depth[0]);
      expect(result.errors).toEqual([]);
    }
    expect((await fixture.evaluate((f: Fixture) => f.dispose()))).toEqual({ errors: [], ownedGpuBytes: 0 });
    expect(pageErrors).toEqual([]);
  } finally {
    await fixture.evaluate((f: Fixture) => f.dispose()).catch(() => {});
    await fixture.dispose();
  }
});

test('double-sided rendering and unlit color respond to the intended controls', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const fixture = await openFixture(page);
  try {
    const backCulled = await fixture.evaluate((f: Fixture) => f.plain('unlit', false, 'back', [2, 2, 2]));
    expect(backCulled.depth[0].surface).toBe(1);
    expect(backCulled.depth[1].surface).toBe(1);
    const backVisible = await fixture.evaluate((f: Fixture) => f.plain('unlit', true, 'back', [2, 2, 2]));
    expectHit(backVisible.depth[0]);
    expectHit(backVisible.depth[1]);

    const unlitBright = await fixture.evaluate((f: Fixture) => f.plain('unlit', false, 'front', [4, 4, 4]));
    expectHit(unlitBright.depth[0]);
    const unlitBrightPixel = await centerPixel(page);
    await fixture.evaluate((f: Fixture) => f.sun([0, 0, 0]));
    const unlitDarkPixel = await centerPixel(page);
    expect(Math.max(...unlitBrightPixel.map((channel, i) => Math.abs(channel - unlitDarkPixel[i])))).toBeLessThanOrEqual(2);

    await fixture.evaluate((f: Fixture) => f.plain('pbr', false, 'front', [4, 4, 4]));
    const pbrBrightPixel = await centerPixel(page);
    await fixture.evaluate((f: Fixture) => f.sun([0, 0, 0]));
    const pbrDarkPixel = await centerPixel(page);
    expect(pbrBrightPixel[0] - pbrDarkPixel[0]).toBeGreaterThan(10);
    expect((await fixture.evaluate((f: Fixture) => f.dispose()))).toEqual({ errors: [], ownedGpuBytes: 0 });
    expect(pageErrors).toEqual([]);
  } finally {
    await fixture.evaluate((f: Fixture) => f.dispose()).catch(() => {});
    await fixture.dispose();
  }
});

test('masked coverage follows deformed geometry and asynchronous replacements keep the final request', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  const fixture = await openFixture(page);
  try {
    const raced = await fixture.evaluate((f: Fixture) => f.replacementRace());
    expectHit(raced.depth[0]);
    expectHole(raced.depth[1]);
    expect(raced.diagnostics.materials).toEqual({ materials: 1, images: 1, imageBytes: 20 });

    const deformed = await fixture.evaluate((f: Fixture) => f.deformedMask());
    expect(deformed.points[0][0]).toBeGreaterThan(-0.5);
    expect(deformed.waveOffset).toBeGreaterThan(0.1);
    expectHit(deformed.depth[0]);
    expectHole(deformed.depth[1]);
    expect(deformed.errors).toEqual([]);

    const disposed = await fixture.evaluate((f: Fixture) => f.disposeDuringPending());
    expect(disposed.outcome).toMatch(/RENDERER_DISPOSED/);
    expect(disposed.ownedGpuBytes).toBe(0);
    expect(disposed.errors).toEqual([]);
    expect(pageErrors).toEqual([]);
  } finally {
    await fixture.evaluate((f: Fixture) => f.dispose()).catch(() => {});
    await fixture.dispose();
  }
});

test('disposal also rejects a replacement after its cached preparation has finished', async ({ page }) => {
  const fixture = await openFixture(page);
  try {
    const result = await fixture.evaluate((f: Fixture) => f.disposeDuringCachedReplacement());
    expect(result.outcome).toMatch(/RENDERER_DISPOSED/);
    expect(result.ownedGpuBytes).toBe(0);
    expect(result.errors).toEqual([]);
  } finally {
    await fixture.evaluate((f: Fixture) => f.dispose()).catch(() => {});
    await fixture.dispose();
  }
});
