#!/usr/bin/env node
// Validates docs/capability-matrix.json against the frozen scope in docs/benchmarks.md and
// docs/roadmap.md, derives each row's status from its gate results and evidence, and prints a
// report. Exits non-zero when the matrix is inconsistent.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const STATUSES = ['pass', 'fail', 'unsupported', 'partial', 'untested'];
const GATE_RESULTS = ['met', 'unmet', 'partly-measured', 'not-measured'];
const EVIDENCE_KINDS = ['functional', 'quality', 'long-run', 'diagnostic-timing', 'acceptance-timing'];
// target: the target adapter in the target browser; historical: earlier hardware; compatibility: a
// candidate or functional-only browser entry. Only target evidence supports gate results or status.
const EVIDENCE_SCOPES = ['target', 'historical', 'compatibility'];
const COMPATIBILITY_ROLES = ['candidate', 'functional-only'];
const ROW_FIELDS = ['id', 'kind', 'subsystem', 'priority', 'phase', 'basis', 'inputScale', 'correctnessReference', 'qualityGate',
  'performanceBudget', 'compatibleCombinations', 'gateResults', 'unsupportedCapabilities', 'status', 'statusReason', 'targetEvidence',
  'historicalEvidence', 'limitations'];

/** Rows of the first Markdown table after `## heading`, as arrays of trimmed cells (header and rule excluded). */
export function markdownTable(markdown, heading) {
  const lines = markdown.split('\n');
  const start = lines.findIndex(line => line.trim() === '## ' + heading || line.trim() === '### ' + heading);
  if (start < 0) throw new Error('MATRIX_DOC: heading not found: ' + heading);
  const rows = [];
  let inTable = false;
  for (const line of lines.slice(start + 1)) {
    if (/^#{2,3} /.test(line)) break;
    if (line.trim().startsWith('|')) { inTable = true; rows.push(line.trim().slice(1, -1).split('|').map(cell => cell.trim())); continue; }
    if (inTable) break;
  }
  return rows.slice(2);
}

function headings(markdown) {
  return new Set(markdown.split('\n').filter(line => /^#{2,3} /.test(line)).map(line => line.replace(/^#{2,3} /, '').trim()));
}

/** Status implied by gate results, unsupported capabilities and evidence; `pass` needs acceptance timing for any budget. */
export function deriveStatus(row, evidence) {
  const results = Object.values(row.gateResults ?? {});
  if (results.includes('unmet')) return 'fail';
  if (row.unsupportedCapabilities?.length) return 'unsupported';
  if (!row.targetEvidence?.length) return 'untested';
  const required = ['functional', ...row.qualityGate, ...row.performanceBudget];
  const allMet = required.every(gate => row.gateResults[gate] === 'met');
  const timed = row.targetEvidence.some(item => evidence[item.ref]?.kind === 'acceptance-timing');
  return allMet && (!row.performanceBudget.length || timed) ? 'pass' : 'partial';
}

export function validateMatrix(matrix, { benchmarks, roadmap, fileExists }) {
  const errors = [];
  const fail = message => errors.push(message);
  if (matrix.schema !== 'real-engine.capability-matrix') fail('schema must be real-engine.capability-matrix');
  if (!Number.isInteger(matrix.version) || matrix.version < 1) fail('version must be a positive integer');
  if (!(matrix.changeLog ?? []).some(entry => entry.version === matrix.version && entry.reason && entry.change)) fail('changeLog needs an entry with change and reason for the current version');
  const target = matrix.targetHardware?.adapter;
  if (target !== 'apple / metal-3') fail('targetHardware.adapter must be apple / metal-3');
  if (STATUSES.some(status => !matrix.statusSemantics?.[status])) fail('statusSemantics must define ' + STATUSES.join(', '));

  const gates = matrix.globalGates ?? [];
  const documented = markdownTable(benchmarks, '初始验收目标');
  if (gates.length !== documented.length) fail(`globalGates has ${gates.length} gates, benchmarks.md has ${documented.length}`);
  documented.forEach(([item, check], i) => {
    if (gates[i]?.item !== item || gates[i]?.check !== check) fail(`globalGates[${i}] must copy benchmarks.md verbatim: ${item}`);
  });
  const gateIds = new Set(gates.map(gate => gate.id));
  if (gateIds.size !== gates.length) fail('globalGates ids must be unique');

  const rows = matrix.rows ?? [];
  const ids = new Set();
  for (const row of rows) {
    if (ids.has(row.id)) fail('duplicate row id ' + row.id);
    ids.add(row.id);
    for (const field of ROW_FIELDS) if (row[field] === undefined) fail(`${row.id}: missing ${field}`);
  }
  for (const [id, action, checks] of markdownTable(benchmarks, '固定场景集')) {
    const row = rows.find(candidate => candidate.id === id);
    if (!row) { fail('missing scene row ' + id); continue; }
    if (row.kind !== 'scene' || row.scene?.action !== action || row.scene?.checks !== checks) fail(`${id}: scene must copy benchmarks.md verbatim`);
  }
  for (const [table, heading] of [['阶段 1', '阶段 1：建立可运行的统一基础'], ['阶段 2', '阶段 2：虚拟几何与世界空间光照']]) {
    for (const [name] of markdownTable(roadmap, heading)) {
      const matches = rows.filter(row => row.kind === 'work-package' && row.source?.table === table && row.source?.name === name);
      if (matches.length !== 1) fail(`${table} work package ${name} needs exactly one row, found ${matches.length}`);
    }
  }
  const sections = headings(roadmap);
  for (const row of rows.filter(candidate => candidate.kind === 'deliverable')) {
    if (!sections.has(row.source?.section)) fail(`${row.id}: roadmap.md has no section ${row.source?.section}`);
  }

  const evidence = matrix.evidence ?? {};
  const browsers = new Map((matrix.browsers ?? []).map(entry => [entry.id, entry]));
  for (const [key, record] of Object.entries(evidence)) {
    if (record.scope === 'target' && record.adapter !== target) fail(`evidence ${key}: target evidence must come from ${target}`);
    if (record.scope === 'historical' && record.adapter === target) fail(`evidence ${key}: historical evidence must not use the target adapter`);
    if (record.scope === 'compatibility') {
      const browser = browsers.get(record.browserId);
      if (!COMPATIBILITY_ROLES.includes(browser?.role)) fail(`evidence ${key}: compatibility evidence must name a ${COMPATIBILITY_ROLES.join(' or ')} browser`);
      else if (record.adapter !== browser.adapter) fail(`evidence ${key}: adapter must be ${browser.adapter} as browser ${browser.id} reports it`);
      if (record.adapter === target) fail(`evidence ${key}: compatibility evidence must not use the target adapter`);
    }
    if (!EVIDENCE_SCOPES.includes(record.scope)) fail(`evidence ${key}: scope must be one of ${EVIDENCE_SCOPES.join(', ')}`);
    if (!EVIDENCE_KINDS.includes(record.kind)) fail(`evidence ${key}: kind must be one of ${EVIDENCE_KINDS.join(', ')}`);
  }
  for (const entry of matrix.browsers ?? []) {
    for (const ref of entry.evidence ?? []) {
      if (evidence[ref]?.scope !== 'compatibility' || evidence[ref].browserId !== entry.id) fail(`browser ${entry.id}: evidence ${ref} is not a compatibility record of this browser`);
    }
  }
  const references = new Set([...Object.keys(matrix.references ?? {}), 'functional']);
  for (const row of rows) {
    if (!['P0', 'P1', 'P2'].includes(row.priority)) fail(`${row.id}: priority must be P0, P1 or P2`);
    if (!Number.isInteger(row.phase) || row.phase < 1 || row.phase > 4) fail(`${row.id}: phase must be 1–4`);
    if (!STATUSES.includes(row.status)) fail(`${row.id}: unknown status ${row.status}`);
    for (const reference of row.correctnessReference ?? []) if (!references.has(reference)) fail(`${row.id}: unknown reference ${reference}`);
    for (const gate of [...(row.qualityGate ?? []), ...(row.performanceBudget ?? [])]) if (!gateIds.has(gate)) fail(`${row.id}: unknown gate ${gate}`);
    const allowed = new Set(['functional', ...(row.qualityGate ?? []), ...(row.performanceBudget ?? [])]);
    const supported = new Set((row.targetEvidence ?? []).flatMap(item => item.gates ?? []));
    for (const [gate, result] of Object.entries(row.gateResults ?? {})) {
      if (!allowed.has(gate)) fail(`${row.id}: gate result for unlisted gate ${gate}`);
      if (!GATE_RESULTS.includes(result)) fail(`${row.id}: gate result ${gate}=${result} is not one of ${GATE_RESULTS.join(', ')}`);
      if (result !== 'not-measured' && !supported.has(gate)) fail(`${row.id}: ${gate}=${result} has no target evidence`);
    }
    for (const item of row.targetEvidence ?? []) {
      if (evidence[item.ref]?.scope !== 'target') fail(`${row.id}: target evidence ${item.ref} is not a target record`);
      for (const spec of item.specs ?? []) if (!fileExists(spec)) fail(`${row.id}: evidence spec ${spec} does not exist`);
    }
    for (const item of row.historicalEvidence ?? []) {
      if (evidence[item.ref]?.scope !== 'historical') fail(`${row.id}: historical evidence ${item.ref} is not a historical record`);
    }
    for (const item of row.compatibilityEvidence ?? []) {
      if (evidence[item.ref]?.scope !== 'compatibility') fail(`${row.id}: compatibility evidence ${item.ref} is not a compatibility record`);
      for (const spec of item.specs ?? []) if (!fileExists(spec)) fail(`${row.id}: evidence spec ${spec} does not exist`);
    }
    if (row.status === 'unsupported' && !(row.unsupportedCapabilities ?? []).length) fail(`${row.id}: unsupported rows must name the missing capability`);
    if (row.status === 'untested' && Object.keys(row.gateResults ?? {}).length) fail(`${row.id}: untested rows cannot carry gate results`);
    const derived = deriveStatus(row, evidence);
    if (derived !== row.status) fail(`${row.id}: status ${row.status} does not match derived ${derived}`);
    if (!row.statusReason) fail(`${row.id}: statusReason is required`);
  }
  return errors;
}

export function summarizeMatrix(matrix) {
  const count = rows => Object.fromEntries(STATUSES.map(status => [status, rows.filter(row => row.status === status).length]));
  const phases = [...new Set(matrix.rows.map(row => row.phase))].sort();
  return {
    version: matrix.version, frozen: matrix.frozen, evidenceAsOf: matrix.evidenceAsOf, evidenceMain: matrix.evidenceMain ?? null,
    target: matrix.targetHardware.adapter,
    total: count(matrix.rows),
    byPhase: Object.fromEntries(phases.map(phase => [phase, count(matrix.rows.filter(row => row.phase === phase))])),
    p0: count(matrix.rows.filter(row => row.priority === 'P0')),
    rows: matrix.rows.map(row => ({ id: row.id, phase: row.phase, priority: row.priority, status: row.status, reason: row.statusReason })),
  };
}

function markdownReport(summary) {
  const header = `能力矩阵 v${summary.version}（冻结 ${summary.frozen}，证据截至 ${summary.evidenceAsOf}${summary.evidenceMain ? `（main ${summary.evidenceMain}）` : ''}，目标 ${summary.target}）`;
  const totals = STATUSES.map(status => `${status} ${summary.total[status]}`).join('，');
  const lines = [header, '', '合计：' + totals, '', '| ID | 阶段 | 优先级 | 状态 | 依据 |', '| --- | --- | --- | --- | --- |'];
  for (const row of summary.rows) lines.push(`| ${row.id} | ${row.phase} | ${row.priority} | ${row.status} | ${row.reason.replaceAll('|', '\\|')} |`);
  return lines.join('\n') + '\n';
}

function main(argv) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const read = path => readFileSync(join(root, path), 'utf8');
  const matrix = JSON.parse(read('docs/capability-matrix.json'));
  const errors = validateMatrix(matrix, { benchmarks: read('docs/benchmarks.md'), roadmap: read('docs/roadmap.md'), fileExists: path => existsSync(join(root, path)) });
  const summary = summarizeMatrix(matrix);
  const markdownAt = argv.indexOf('--markdown');
  if (markdownAt >= 0) {
    if (!argv[markdownAt + 1]) throw new Error('--markdown needs an output path');
    writeFileSync(resolve(argv[markdownAt + 1]), markdownReport(summary));
  }
  if (argv.includes('--json')) console.log(JSON.stringify({ errors, ...summary }, null, 2));
  else {
    process.stdout.write(markdownReport(summary));
    for (const error of errors) console.error('MATRIX_INVALID ' + error);
  }
  return errors.length ? 1 : 0;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
