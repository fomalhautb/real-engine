import { chromium } from '@playwright/test';
import { mkdir, readFile, writeFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

// Explicit local diagnostic entry point. The caller supplies a bounded case
// manifest extracted from a prior capture; no downloads or asset builds occur.
const [inputPath, output] = process.argv.slice(2);
if (!inputPath || !output || process.argv.length !== 4) {
  throw Error('Usage: node scripts/capture-sponza-anchor-reference.mjs input.json output-directory');
}
if ((await stat(inputPath)).size > 2 * 1024 ** 2) throw Error('SPONZA_REFERENCE_INPUT_BYTES');
const input = JSON.parse(await readFile(inputPath, 'utf8'));
// Fail malformed links before launching a browser (including its default demo
// renderer). The fixture repeats validation and checks the actual ray wire form.
if (input.visibilityLinks !== undefined) {
  const links = input.visibilityLinks, labels = new Set();
  if (!Array.isArray(links) || links.length > 256) throw Error('SPONZA_REFERENCE_LINK_INPUT');
  for (const link of links) {
    if (!link || typeof link.label !== 'string' || !link.label.length || link.label.length > 128 || labels.has(link.label)
      || [link.origin, link.destination].some(v => !Array.isArray(v) || v.length !== 3
        || v.some(value => !Number.isFinite(value) || !Number.isFinite(Math.fround(value))))) throw Error('SPONZA_REFERENCE_LINK_INPUT');
    labels.add(link.label);
    const distance = Math.hypot(...link.destination.map((value, axis) => value - link.origin[axis]));
    if (!(distance > 0.0002) || distance > 1000 || !(Math.fround(distance - 0.0001) > Math.fround(0.0001))) {
      throw Error('SPONZA_REFERENCE_LINK_DISTANCE');
    }
  }
}
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--force-high-performance-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 1320, height: 800 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(600000);
  const pageErrors = [], writes = [];
  page.on('pageerror', error => { pageErrors.push(error.message); console.log('PAGE_ERROR ' + error.message); });
  page.on('console', message => {
    const text = message.text();
    if (!text.startsWith('SPONZA_REFERENCE_PROGRESS ')) return;
    const event = JSON.parse(text.slice('SPONZA_REFERENCE_PROGRESS '.length));
    if (event.phase === 'reference-batch') {
      writes.push(writeFile(join(output, `batch-${event.data.batch}.json`), JSON.stringify(event.data, null, 2)));
      console.log(JSON.stringify({ phase: event.phase, elapsedMs: event.elapsedMs,
        batch: event.data.batch, seed: event.data.seed, samplesPerCase: event.data.samplesPerCase }));
    } else console.log(text);
  });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async input => {
    window.__REAL_ENGINE__.pause(true); window.__REAL_ENGINE__.dispose();
    const { captureIntelSponzaAnchorReference } = await import('/tests/browser/fixtures/intel-sponza-anchor-reference.ts');
    return captureIntelSponzaAnchorReference(input);
  }, input);
  await Promise.all(writes);
  await writeFile(join(output, 'report.json'), JSON.stringify({ ...result, input, pageErrors }, null, 2));
  if (pageErrors.length || result.errors.length || result.afterDispose !== 0 || result.info.adapterInfo.isFallbackAdapter) {
    throw Error('SPONZA_REFERENCE_VALIDATION');
  }
  const expectedLinks = input.visibilityLinks ?? [], links = result.reference.visibilityLinks;
  if ((links?.rows.length ?? 0) !== expectedLinks.length || (expectedLinks.length && (
    links.snapshot.frame !== result.reference.snapshot.frame || links.snapshot.materialRevision !== result.reference.snapshot.materialRevision
    || links.rows.some((row, index) => row.label !== expectedLinks[index].label || row.blocked !== (row.hit !== null)
      || row.origin.some((value, axis) => value !== expectedLinks[index].origin[axis])
      || row.destination.some((value, axis) => value !== expectedLinks[index].destination[axis])
      || !Number.isFinite(row.distance) || !(row.distance > 0.0002) || row.distance > 1000
      || row.tMin !== 0.0001 || Math.abs(row.tMax - (row.distance - 0.0001)) > 1e-12
      || (row.hit && (!Number.isFinite(row.hit.distance) || row.hit.distance < Math.fround(row.tMin) || row.hit.distance > Math.fround(row.tMax))))))) {
    throw Error('SPONZA_REFERENCE_VISIBILITY_LINK_VALIDATION');
  }
  console.log(JSON.stringify({ adapter: result.info.adapterInfo, samplesPerCase: result.reference.budget.samplesPerCase,
    rows: result.reference.rows.map(row => ({ label: row.label, luminance: row.contributions.total.luminance })),
    visibilityLinks: links ? { total: links.rows.length, blocked: links.rows.filter(link => link.blocked).length } : null,
    afterDispose: result.afterDispose }));
} finally { await browser.close(); }
