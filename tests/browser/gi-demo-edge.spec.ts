import { expect, test } from '@playwright/test';
import { lookAt, multiply, perspective, transformPoint } from '../../src/math';

type Vec3 = [number, number, number];
const wallPoints: Vec3[] = [
  [3.2, 2, 3.45],
  [3.2, 2, 3.60],
  [3.2, 2, 3.68],
];

test('GI lights the blue inner wall through its visible front edge', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1024, height: 768 });
  try {
    await page.goto('/gi.html');
    await page.waitForFunction(() => window.__GI_DEMO__?.renderer && window.__GI_DEMO__.renderer.stats.frames > 0
      || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    await page.evaluate(() => window.__GI_DEMO__!.pause(true));
    await page.keyboard.press('h');
    await expect(page.locator('#gi-app')).toHaveClass(/ui-hidden/);

    const fixed = await page.evaluate(() => {
      const demo = window.__GI_DEMO__!;
      demo.setMultiBounce(false);
      demo.setMultiBounce(true); // The handler renders the first feedback-free frame.
      const camera = demo.snapshot().camera;
      for (let frame = 1; frame < 12; frame++) {
        demo.renderer!.render(camera.eye, camera.target as [number, number, number]);
      }
      const snapshot = demo.snapshot();
      return { camera: snapshot.camera, stats: snapshot.stats!, gi: snapshot.diffuseGi!,
        grid: snapshot.probeGrid, errors: snapshot.errors };
    });

    const eye = fixed.camera.eye as Vec3;
    const target = fixed.camera.target as Vec3;
    const width = fixed.stats.width, height = fixed.stats.height;
    const projection = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150),
      lookAt(eye, target, [0, 1, 0]));
    const projected = wallPoints.map(world => {
      const ndc = transformPoint(projection, world);
      return { world, ndc, pixel: {
        x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
        y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5),
      } };
    });
    for (const { ndc, pixel } of projected) {
      expect(Math.abs(ndc[0])).toBeLessThan(1);
      expect(Math.abs(ndc[1])).toBeLessThan(1);
      expect(ndc[2]).toBeGreaterThan(0);
      expect(ndc[2]).toBeLessThan(1);
      expect(pixel.x).toBeGreaterThanOrEqual(0);
      expect(pixel.x).toBeLessThan(width);
      expect(pixel.y).toBeGreaterThanOrEqual(0);
      expect(pixel.y).toBeLessThan(height);
    }

    const measured = await page.evaluate(async ({ eye, points, pixels }) => {
      const renderer = window.__GI_DEMO__!.renderer!;
      const rays = points.map(point => ({ origin: eye,
        direction: [point[0] - eye[0], point[1] - eye[1], point[2] - eye[2]] as Vec3 }));
      const hits = (await renderer.traceRays(rays)).hits;
      const hdr = await renderer.probeHdr(pixels);
      return { hits, hdr, frame: renderer.stats.frames, errors: [...renderer.errors] };
    }, { eye, points: wallPoints, pixels: projected.map(value => value.pixel) });

    // Persist the real canvas image even if a later numerical assertion fails.
    const imagePath = testInfo.outputPath('gi-blue-wall-front-edge.png');
    await page.locator('#viewport').screenshot({ path: imagePath });
    await testInfo.attach('gi-blue-wall-front-edge', { path: imagePath, contentType: 'image/png' });
    console.log('GI_DEMO_EDGE_GPU_DIAGNOSTIC ' + JSON.stringify({
      frame: measured.frame, camera: fixed.camera, grid: fixed.grid,
      projected, hdr: measured.hdr, hits: measured.hits,
      historyUpdates: fixed.gi.historyUpdates,
    }));

    expect(fixed.gi.multiBounce).toBe(true);
    expect(fixed.gi.historyUpdates).toBe(12);
    expect(measured.frame).toBe(fixed.stats.frames);
    for (let i = 0; i < wallPoints.length; i++) {
      const hit = measured.hits[i];
      expect(hit, `world point ${i} must be visible on the blue wall`).not.toBeNull();
      expect(hit!.batch).toBe('blue-wall');
      expect(hit!.entityId).toBe(4);
      expect(hit!.shading).toBe('pbr');
      expect(hit!.normal[0]).toBeLessThan(-0.9);
      expect(hit!.baseColor[2]).toBeCloseTo(0.72, 2);
      for (let axis = 0; axis < 3; axis++) {
        expect(Math.abs(hit!.position[axis] - wallPoints[i][axis])).toBeLessThan(0.02);
      }
      expect(measured.hdr[i][2], `blue wall HDR at z=${wallPoints[i][2]}`).toBeGreaterThan(0.003);
    }
    expect(measured.hdr[2][2] / measured.hdr[0][2]).toBeGreaterThan(0.2);
    expect(fixed.errors).toEqual([]);
    expect(measured.errors).toEqual([]);
    expect(pageErrors).toEqual([]);
  } finally {
    await page.evaluate(() => window.__GI_DEMO__?.dispose()).catch(() => {});
  }
});
