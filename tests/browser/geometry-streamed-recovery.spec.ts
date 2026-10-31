import { writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import type { runStreamedGalleryFixture, StreamedGalleryFixtureOptions } from './fixtures/geometry-streamed-gallery';

// Recovery-speed A/B, not a gate: GEOMETRY_RECOVERY_ABBA=1 enables it. Interval
// admission (A) and work-conserving admission (B) run in A,B,B,A order on fresh
// renderers in one page, without probes during transitions (checkEvery 0), so
// paired differences are less sensitive to drift on a contended machine.
test.skip(!process.env.GEOMETRY_RECOVERY_ABBA, 'set GEOMETRY_RECOVERY_ABBA=1 to measure capacity recovery A/B');

test('capacity admission recovery after teleports and fast turns (ABBA)', async ({ page }, testInfo) => {
  test.setTimeout(1_200_000);
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const objects = Number(process.env.GEOMETRY_RECOVERY_OBJECTS ?? 16);
  const runs: unknown[] = [];
  for (const admission of ['interval', 'work-conserving', 'work-conserving', 'interval'] as const) {
    const options: StreamedGalleryFixtureOptions = { scene: { objects, around: 96, across: 32, admission }, width: 640, height: 400,
      checkEvery: 0, walk: false };
    const r = await page.evaluate(async fixtureOptions => {
      const fixture = await import('/tests/browser/fixtures/' + 'geometry-streamed-gallery.ts') as { runStreamedGalleryFixture: typeof runStreamedGalleryFixture };
      return fixture.runStreamedGalleryFixture(fixtureOptions);
    }, options);
    expect(r.errors).toEqual([]);
    const budget = r.budget as { migrations?: number; earlyRounds?: number };
    runs.push({ admission, adapter: r.adapter, phases: r.phases.map(phase => ({ label: phase.label, frames: phase.framesToLastChange,
      ms: phase.msToLastChange, migrations: phase.migrations })), migrations: budget.migrations ?? null, earlyRounds: budget.earlyRounds ?? null,
      backpressureFrames: r.backpressureFrames });
  }
  writeFileSync(testInfo.outputPath('streamed-recovery-abba.json'), JSON.stringify(runs, null, 2));
  console.log('GEOMETRY_STREAMED_RECOVERY', JSON.stringify(runs));
});
