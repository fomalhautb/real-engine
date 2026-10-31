import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

interface Row {
  id: string; kind: string; priority: string; phase: number; status: string; qualityGate: string[]; performanceBudget: string[];
  gateResults: Record<string, string>; unsupportedCapabilities: string[]; targetEvidence: { ref: string, specs: string[], gates: string[] }[];
  historicalEvidence: { ref: string }[];
}
interface Matrix {
  version: number; targetHardware: { adapter: string }; globalGates: { id: string, item: string, check: string }[];
  evidence: Record<string, { scope: string, adapter: string, kind: string }>; rows: Row[];
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
  });

  it('refuses a pass or a measured result that the evidence does not support', () => {
    const promoted = copy();
    row(promoted, 'F01').status = 'pass';
    expect(validateMatrix(promoted, docs)).toContain('F01: status pass does not match derived partial');
    const unsupported = copy();
    row(unsupported, 'G02').gateResults = { functional: 'met' };
    expect(validateMatrix(unsupported, docs)).toContain('G02: functional=met has no target evidence');
    const hidden = copy();
    row(hidden, 'G03').gateResults = { functional: 'partly-measured' };
    expect(validateMatrix(hidden, docs)).toContain('G03: status fail does not match derived partial');
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
