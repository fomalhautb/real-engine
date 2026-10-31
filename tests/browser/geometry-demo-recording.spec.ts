import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';

// Demo footage, not a gate: GEOMETRY_RECORD_DIR=<dir> enables it. Each clip is
// a fresh page on one fixed trajectory with fixed quality parameters; the
// split clips show normal material (left) and one diagnostic view (right) in
// the same frame. The .webm keeps page loading; FFMPEG (default
// /opt/homebrew/bin/ffmpeg) writes an H.264 .mp4 that starts at the trajectory.
const directory = process.env.GEOMETRY_RECORD_DIR;
const ffmpeg = process.env.FFMPEG ?? '/opt/homebrew/bin/ffmpeg';
type Clip = { name: string; trajectory: 'walk' | 'zoom' | 'orbit' | 'turn' | 'teleport'; seconds: number; debug: 'material' | 'clusters' | 'levels' | 'residency'; split: number; ui: boolean };
// Walk clips stop before the default three-room walk reaches the far wall (22.2 s).
const clips: Clip[] = [
  { name: 'walk-material', trajectory: 'walk', seconds: 20, debug: 'material', split: 0, ui: false },
  { name: 'walk-material-vs-residency', trajectory: 'walk', seconds: 20, debug: 'residency', split: 0.5, ui: false },
  { name: 'walk-clusters', trajectory: 'walk', seconds: 20, debug: 'clusters', split: 0, ui: false },
  { name: 'zoom-material-vs-levels', trajectory: 'zoom', seconds: 16, debug: 'levels', split: 0.5, ui: false },
  { name: 'orbit-material-vs-clusters', trajectory: 'orbit', seconds: 13, debug: 'clusters', split: 0.5, ui: false },
  { name: 'turn-material-vs-residency', trajectory: 'turn', seconds: 9, debug: 'residency', split: 0.5, ui: false },
  { name: 'teleport-material-vs-residency', trajectory: 'teleport', seconds: 10, debug: 'residency', split: 0.5, ui: false },
  { name: 'walk-panel-residency', trajectory: 'walk', seconds: 20, debug: 'residency', split: 0, ui: true },
];

test.skip(!directory, 'set GEOMETRY_RECORD_DIR to record geometry demo footage');

test('record streamed geometry demo trajectories', async ({ browser }) => {
  test.setTimeout(clips.length * 120_000);
  mkdirSync(directory!, { recursive: true });
  const query = process.env.GEOMETRY_RECORD_QUERY ?? 'scene=streamed';
  const log: unknown[] = [];
  for (const clip of clips) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, recordVideo: { dir: directory!, size: { width: 1280, height: 720 } } });
    const page = await context.newPage();
    // The video starts with the page; the trajectory starts after loading.
    const videoStarted = Date.now();
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`/geometry.html?${query}`);
    await page.waitForFunction(() => window.__GEOMETRY_DEMO__ || !document.getElementById('error')!.hidden, null, { timeout: 120_000 });
    await expect(page.locator('#error')).toBeHidden();
    await page.evaluate(({ trajectory, debug, split, ui }) => {
      const demo = window.__GEOMETRY_DEMO__!;
      demo.pause(true); demo.setDebug(debug); demo.setSplit(split); demo.seek(trajectory, 0);
      if (!ui) document.getElementById('geometry-app')!.classList.add('ui-hidden');
      demo.pause(false);
    }, clip);
    const started = Date.now();
    await page.waitForTimeout(clip.seconds * 1000);
    const snapshot = await page.evaluate(() => {
      const value = window.__GEOMETRY_DEMO__!.snapshot();
      return { frames: value.stats?.frames, adapter: value.adapter?.adapter, timing: value.adapter?.timing, accounting: value.accounting,
        sceneOptions: value.sceneOptions, controls: value.controls, streams: value.streams, rafIntervalMs: value.rafIntervalMs, errors: value.errors };
    });
    const video = page.video();
    await context.close();
    const webm = await video!.path(), base = join(directory!, clip.name), leadSeconds = (started - videoStarted) / 1000;
    renameSync(webm, base + '.webm');
    if (existsSync(ffmpeg)) {
      execFileSync(ffmpeg, ['-y', '-loglevel', 'error', '-i', base + '.webm', '-ss', leadSeconds.toFixed(2), '-t', String(clip.seconds),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', base + '.mp4']);
    }
    log.push({ clip, leadSeconds, wallMs: Date.now() - started, snapshot, pageErrors: errors });
    expect(errors).toEqual([]);
  }
  writeFileSync(join(directory!, 'recordings.json'), JSON.stringify({ query, recordedAt: new Date().toISOString(), log }, null, 2));
});
