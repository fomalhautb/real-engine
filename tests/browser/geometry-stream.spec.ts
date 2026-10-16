import { expect, test } from '@playwright/test';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { gzipSync } from 'node:zlib';
import type { prepareGeometryStreamFixture, runGeometryStreamFixture } from './fixtures/geometry-stream';

type Prepared = Awaited<ReturnType<typeof prepareGeometryStreamFixture>>;
type Result = Awaited<ReturnType<typeof runGeometryStreamFixture>>;

test('HTTP pages feed a bounded CPU cache and atomic reference cut with hash retry', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const prepared: Prepared = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-stream.ts') as {
      prepareGeometryStreamFixture: typeof prepareGeometryStreamFixture;
    };
    return fixture.prepareGeometryStreamFixture();
  });
  expect(prepared.pageCount).toBeGreaterThan(prepared.rootPages.length);
  expect(prepared.initialRequestedPages.length).toBeGreaterThan(0);
  const failedPageId = prepared.initialRequestedPages.find(id => !prepared.rootPages.includes(id));
  expect(failedPageId).toBeDefined();
  const payloads = new Map(prepared.pages.map(entry => [entry.id, entry]));
  const requests: number[] = [], attempts = new Map<number, number>();
  // A real local server is required: intercepted fulfill() bodies bypass the
  // browser's normal HTTP decompressor. CORS also hides Content-Encoding here.
  const server = createServer((request, response) => {
    const name = new URL(request.url!, 'http://127.0.0.1').pathname.split('/').at(-1)!;
    const match = /^page-(\d+)\.bin$/.exec(name);
    const id = match ? Number(match[1]) : -1;
    const payload = payloads.get(id);
    if (!payload) { response.writeHead(404); response.end(); return; }
    const attempt = attempts.get(id) ?? 0;
    attempts.set(id, attempt + 1);
    requests.push(id);
    const bytes = Buffer.from(payload.base64, 'base64');
    if (id === failedPageId && attempt === 0) bytes[bytes.length - 1] ^= 1;
    // Exercise real browser HTTP decompression as well as the meshopt page codec.
    const compressed = id === prepared.rootPages[0] || id === failedPageId;
    const body = compressed ? gzipSync(bytes) : bytes;
    response.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Access-Control-Allow-Origin': '*',
      'Content-Length': String(body.byteLength), ...(compressed ? { 'Content-Encoding': 'gzip' } : {}) });
    response.end(body);
  });
  let result: Result;
  try {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject); server.listen(0, '127.0.0.1', resolve);
    });
    const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
    result = await page.evaluate(async ({ id, baseUrl }) => {
      const fixture = await import('/tests/browser/fixtures/' + 'geometry-stream.ts') as {
        runGeometryStreamFixture: typeof runGeometryStreamFixture;
      };
      return fixture.runGeometryStreamFixture(id, baseUrl);
    }, { id: failedPageId!, baseUrl });
  } finally {
    server.closeAllConnections();
    if (server.listening) await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
  expect(result.mode).toMatch(/CPU reference.*no GPU upload/);
  expect(result.pageCount).toBe(prepared.pageCount);
  expect(result.rootPages).toEqual(prepared.rootPages);
  expect(result.rootState.residentPages).toEqual([...prepared.rootPages].sort((a, b) => a - b));
  expect(result.rootState.pinnedPages).toEqual([...prepared.rootPages].sort((a, b) => a - b));
  expect(result.rootState.requestsStarted).toBe(prepared.rootPages.length);
  expect(result.rootPlan.selectedPages.every(id => prepared.rootPages.includes(id))).toBe(true);
  expect(result.rootPlan.requestedPages.map(entry => entry.pageId)).toEqual(prepared.initialRequestedPages);
  expect(result.rootPlan.fallbackGroups.length).toBeGreaterThan(0);
  expect(result.partialState.failedPages).toContainEqual({ pageId: failedPageId, message: `GEOMETRY_STREAM_HASH: ${failedPageId}` });
  expect(result.partialState.residentPages).toContain(prepared.rootPages[0]);
  expect(result.partialPlan.fallbackGroups.length).toBeGreaterThan(0);
  expect(result.partialPlan.requestedPages.map(entry => entry.pageId)).toContain(failedPageId);
  expect(result.atomicFallback).toBe(true);
  expect(result.fullState.failedPages).toEqual([]);
  expect(result.fullState.residentPages).toHaveLength(prepared.pageCount);
  expect(result.fullPlan.fallbackGroups).toEqual([]);
  expect(result.fullPlan.refinedGroups.length).toBeGreaterThan(0);
  expect(result.fullPlan.selectedClusters).toEqual(prepared.leafClusters);
  expect(result.fullTriangles).toBe(prepared.sourceTriangles);
  expect(result.fullTriangles).toBeGreaterThan(result.rootTriangles);
  expect(result.selectedPagesAvailable).toBe(true);
  expect(requests.slice(0, prepared.rootPages.length).sort((a, b) => a - b))
    .toEqual([...prepared.rootPages].sort((a, b) => a - b));
  expect(attempts.get(failedPageId!)).toBe(2);
  expect(result.disposedState).toMatchObject({ residentBytes: 0, reservedBytes: 0, inFlightBytes: 0,
    activeRequests: 0, queuedRequests: 0, residentPages: [], disposed: true, gpuResidency: false });
  expect(pageErrors).toEqual([]);
  console.log('GEOMETRY_STREAM_CPU_REFERENCE_DIAGNOSTIC', JSON.stringify({
    pages: result.pageCount, clusters: prepared.clusterCount, roots: result.rootPages,
    rootSelectedPages: result.rootPlan.selectedPages, partialFallbackGroups: result.partialPlan.fallbackGroups,
    fineSelectedPages: result.fullPlan.selectedPages, failedPageId, failedAttempts: attempts.get(failedPageId!),
    rootTriangles: result.rootTriangles, fullTriangles: result.fullTriangles,
    ownedCpuBytesAfterDispose: result.disposedState.residentBytes,
  }));
});
