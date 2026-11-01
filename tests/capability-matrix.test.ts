import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

interface Row {
  id: string; kind: string; priority: string; phase: number; status: string; qualityGate: string[]; performanceBudget: string[];
  gateResults: Record<string, string>; unsupportedCapabilities: string[]; targetEvidence: { ref: string, specs: string[], gates: string[] }[];
  historicalEvidence: { ref: string }[]; compatibilityEvidence?: { ref: string, specs: string[] }[];
}
interface Matrix {
  version: number; targetHardware: { adapter: string }; globalGates: { id: string, item: string, check: string }[];
  browsers: { id: string, role: string, adapter: string, evidence?: string[] }[];
  evidence: Record<string, { scope: string, adapter: string, kind: string, browserId?: string }>; rows: Row[];
}
interface Docs { benchmarks: string, roadmap: string, fileExists: (path: string) => boolean }

const { validateMatrix, deriveStatus, summarizeMatrix } = await import(pathToFileURL(resolve('scripts/capability-report.mjs')).href) as {
  validateMatrix: (matrix: Matrix, docs: Docs) => string[];
  deriveStatus: (row: Row, evidence: Matrix['evidence']) => string;
  summarizeMatrix: (matrix: Matrix) => { total: Record<string, number> };
};

const matrix = JSON.parse(readFileSync(resolve('docs/capability-matrix.json'), 'utf8')) as Matrix;
const docs: Docs = { benchmarks: readFileSync(resolve('docs/benchmarks.md'), 'utf8'), roadmap: readFileSync(resolve('docs/roadmap.md'), 'utf8'),
  fileExists: path => existsSync(resolve(path)) };
const copy = () => structuredClone(matrix);
const row = (target: Matrix, id: string) => target.rows.find(candidate => candidate.id === id)!;

describe('capability matrix', () => {
  it('matches the frozen scope, copies gates verbatim and derives every status from target evidence', () => {
    expect(validateMatrix(matrix, docs)).toEqual([]);
    expect(matrix.targetHardware.adapter).toBe('apple / metal-3');
    const summary = summarizeMatrix(matrix);
    expect(Object.values(summary.total).reduce((sum, count) => sum + count, 0)).toBe(matrix.rows.length);
    for (const id of ['F01', 'F02', 'F03', 'F04', 'F05']) expect([row(matrix, id).priority, row(matrix, id).phase]).toEqual(['P0', 2]);
    expect(row(matrix, 'F06').phase).toBe(4);
    for (const candidate of matrix.rows.filter(entry => entry.status === 'untested')) expect(candidate.targetEvidence).toEqual([]);
  });

  it('rejects loosened gates, dropped scenes and evidence from the wrong hardware', () => {
    const loosened = copy();
    loosened.globalGates[1].check = loosened.globalGates[1].check.replace('33.3', '40');
    expect(validateMatrix(loosened, docs)).toContain('globalGates[1] must copy benchmarks.md verbatim: 帧时间');
    const dropped = copy();
    dropped.rows = dropped.rows.filter(entry => entry.id !== 'F05');
    expect(validateMatrix(dropped, docs)).toContain('missing scene row F05');
    const relabelled = copy();
    relabelled.evidence['hist-nvidia-stability-20261014'].adapter = 'apple / metal-3';
    expect(validateMatrix(relabelled, docs)).toContain('evidence hist-nvidia-stability-20261014: historical evidence must not use the target adapter');
    const untyped = copy();
    untyped.evidence['apple-benchmark-diagnostic-20261031'].kind = 'timing';
    expect(validateMatrix(untyped, docs)).toContain(
      'evidence apple-benchmark-diagnostic-20261031: kind must be one of functional, quality, long-run, diagnostic-timing, acceptance-timing');
  });

  it('refuses a pass or a measured result that the evidence does not support', () => {
    const promoted = copy();
    row(promoted, 'F01').status = 'pass';
    expect(validateMatrix(promoted, docs)).toContain('F01: status pass does not match derived partial');
    const unsupported = copy();
    row(unsupported, 'G02').gateResults = { functional: 'met' };
    expect(validateMatrix(unsupported, docs)).toContain('G02: functional=met has no target evidence');
    const hidden = copy();
    const failing = hidden.rows.find(entry => entry.status === 'fail' && !entry.unsupportedCapabilities.length)!;
    failing.gateResults = Object.fromEntries(Object.entries(failing.gateResults).map(([gate, result]) => [gate, result === 'unmet' ? 'partly-measured' : result]));
    expect(validateMatrix(hidden, docs)).toContain(`${failing.id}: status fail does not match derived partial`);
    const unmet = copy();
    row(unmet, 'F01').gateResults = { ...row(unmet, 'F01').gateResults, functional: 'unmet' };
    expect(validateMatrix(unmet, docs)).toContain('F01: status partial does not match derived fail');
  });

  it('keeps candidate-browser results as compatibility evidence that never counts for the target', () => {
    const base = copy();
    base.evidence.webkit = { scope: 'compatibility', browserId: 'playwright-webkit', adapter: 'apple / apple', kind: 'functional' };
    expect(validateMatrix(base, docs)).toEqual([]);
    const relabelled = structuredClone(base);
    relabelled.evidence.webkit.adapter = 'apple / metal-3';
    expect(validateMatrix(relabelled, docs)).toEqual(expect.arrayContaining([
      'evidence webkit: adapter must be apple / apple as browser playwright-webkit reports it',
      'evidence webkit: compatibility evidence must not use the target adapter']));
    const targetBrowser = structuredClone(base);
    targetBrowser.evidence.webkit.browserId = 'chrome-macos-apple';
    expect(validateMatrix(targetBrowser, docs)).toContain('evidence webkit: compatibility evidence must name a candidate or functional-only browser');
    const promoted = structuredClone(base);
    row(promoted, 'P1-runtime').targetEvidence.push({ ref: 'webkit', specs: [], gates: ['functional'] });
    expect(validateMatrix(promoted, docs)).toContain('P1-runtime: target evidence webkit is not a target record');
    const misfiled = structuredClone(base);
    row(misfiled, 'P1-runtime').compatibilityEvidence = [{ ref: 'apple-main-suite-20261031', specs: [] }];
    expect(validateMatrix(misfiled, docs)).toContain('P1-runtime: compatibility evidence apple-main-suite-20261031 is not a compatibility record');
    const crossed = structuredClone(base);
    crossed.browsers.find(entry => entry.id === 'safari-26-macos')!.evidence = ['webkit'];
    expect(validateMatrix(crossed, docs)).toContain('browser safari-26-macos: evidence webkit is not a compatibility record of this browser');
    const untested = structuredClone(base.rows.find(entry => entry.status === 'untested')!);
    untested.compatibilityEvidence = [{ ref: 'webkit', specs: [] }];
    expect(deriveStatus(untested, base.evidence)).toBe('untested');
  });

  it('needs uncontended acceptance timing before a row with a budget can pass', () => {
    const f04 = structuredClone(row(matrix, 'F04'));
    f04.gateResults = { functional: 'met', 'media-image': 'met', 'media-resources': 'met' };
    expect(deriveStatus(f04, matrix.evidence)).toBe('partial');
    const evidence = { ...matrix.evidence, timed: { scope: 'target', adapter: 'apple / metal-3', kind: 'acceptance-timing' } };
    f04.targetEvidence = [...f04.targetEvidence, { ref: 'timed', specs: [], gates: ['media-resources'] }];
    expect(deriveStatus(f04, evidence)).toBe('pass');
  });
});
