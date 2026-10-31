import { fluctuation, gate, ghostArea, leakSummary, luminance, mean, normalizedMae, relativeChange, responseTime,
  type Gate } from '../../src/gi-acceptance';
import type { runJointDynamics, runJointLeak, runJointSteadyState } from './fixtures/gi-joint-acceptance';

type Dynamics = Awaited<ReturnType<typeof runJointDynamics>>;
type Phase = Dynamics['phases'][number];
type Steady = Awaited<ReturnType<typeof runJointSteadyState>>;
type Leak = Awaited<ReturnType<typeof runJointLeak>>;

/** Frozen harness constants (linear HDR before exposure; EV 0). */
export const JOINT_ANALYSIS = Object.freeze({ fps: 30, finalWindow: 10, darkThreshold: 0.005, maeEpsilon: 1e-3,
  negligibleChange: 0.02, ghostFraction: 0.1, ghostAbsolute: 0.02, stabilizeBand: 0.03 });

function set(phase: Phase, name: string) {
  const found = phase.sets.find(item => item.name === name);
  if (!found) throw new Error('JOINT_SET_MISSING ' + phase.name + ' ' + name);
  return found;
}
export function roiSeries(phase: Phase, name: string): number[] {
  const { start, count } = set(phase, name);
  return phase.frames.map(frame => mean(frame.luminance.slice(start, start + count)));
}
function pixelAt(phase: Phase, name: string, index: number): number[] {
  const { start, count } = set(phase, name);
  return phase.frames[index].luminance.slice(start, start + count);
}
function pixelMean(phase: Phase, name: string, from: number, to: number): number[] {
  const rows = phase.frames.slice(from, to).map((_, offset) => pixelAt(phase, name, from + offset));
  return rows[0].map((_, i) => mean(rows.map(row => row[i])));
}

/** Ghost area at `after` frames following `eventIndex` (frame 1 = first after the event). */
function ghost(phase: Phase, name: string, eventIndex: number, after: number, baselineFrom: number): number | null {
  const index = eventIndex + after - 1;
  if (index >= phase.frames.length - JOINT_ANALYSIS.finalWindow) return null;
  const final = pixelMean(phase, name, phase.frames.length - JOINT_ANALYSIS.finalWindow, phase.frames.length);
  const initial = eventIndex > baselineFrom ? pixelMean(phase, name, baselineFrom, eventIndex) : final;
  const absolute = Math.max(1e-4, JOINT_ANALYSIS.ghostAbsolute * mean(final.map(Math.abs)));
  return ghostArea(pixelAt(phase, name, index), initial, final, { fraction: JOINT_ANALYSIS.ghostFraction, absolute });
}

/** Frames after `eventIndex` until a series stays within `band` of its final value. */
function stabilizeFrames(series: readonly number[], eventIndex: number, band: number): number | null {
  const final = mean(series.slice(series.length - JOINT_ANALYSIS.finalWindow));
  let last = eventIndex - 1;
  for (let i = eventIndex; i < series.length; i++) if (Math.abs(series[i] - final) > band * Math.max(Math.abs(final), 1e-4)) last = i;
  return last + 1 <= series.length - JOINT_ANALYSIS.finalWindow ? last + 1 - eventIndex + 1 : null;
}

function phaseOf(run: Dynamics, name: string): Phase {
  const phase = run.phases.find(item => item.name === name);
  if (!phase) throw new Error('JOINT_PHASE_MISSING ' + name);
  return phase;
}

export function analyzeDynamics(run: Dynamics) {
  const response: Record<string, Record<string, ReturnType<typeof responseTime> & { ghost15: number | null;
    ghost30: number | null }>> = {};
  for (const name of ['sunOff', 'sunOn', 'panelOff', 'panelOn', 'doorClose', 'doorOpen']) {
    const phase = phaseOf(run, name), eventIndex = phase.eventIndex!;
    response[name] = {};
    for (const item of phase.sets.filter(entry => entry.kind !== 'ghost')) {
      const series = roiSeries(phase, item.name);
      const peak = Math.max(...series.map(Math.abs));
      response[name][item.name] = { ...responseTime(series, eventIndex, { finalWindow: JOINT_ANALYSIS.finalWindow,
        baselineWindow: eventIndex, minimumChange: JOINT_ANALYSIS.negligibleChange * peak }),
      ghost15: ghost(phase, item.name, eventIndex, 15, 0), ghost30: ghost(phase, item.name, eventIndex, 30, 0) };
    }
  }
  const stabilityPhase = phaseOf(run, 'stability');
  const stability = Object.fromEntries(stabilityPhase.sets.map(item =>
    [item.name, fluctuation(roiSeries(stabilityPhase, item.name), JOINT_ANALYSIS.darkThreshold)]));
  const mover = phaseOf(run, 'mover'), moverStop = mover.stopIndex!;
  const remove = phaseOf(run, 'removeForeground');
  const teleport = phaseOf(run, 'teleport'), away = phaseOf(run, 'away'), panelOn = phaseOf(run, 'panelOn');
  const panelMove = phaseOf(run, 'panelMove'), panelStop = panelMove.stopIndex!;
  const finalOf = (phase: Phase, name: string) => mean(roiSeries(phase, name).slice(-JOINT_ANALYSIS.finalWindow));
  const revealed = remove.sets.some(item => item.name === 'foreground.revealed');
  const t01 = {
    moverPatchGhost: { after15: ghost(mover, 'mover.patch', moverStop, 15, 0), after30: ghost(mover, 'mover.patch', moverStop, 30, 0),
      stabilizeFrames: stabilizeFrames(roiSeries(mover, 'mover.patch'), moverStop, JOINT_ANALYSIS.stabilizeBand) },
    revealed: revealed ? { after1: ghost(remove, 'foreground.revealed', remove.eventIndex!, 1, 0),
      after15: ghost(remove, 'foreground.revealed', remove.eventIndex!, 15, 0),
      after30: ghost(remove, 'foreground.revealed', remove.eventIndex!, 30, 0),
      stabilizeFrames: stabilizeFrames(roiSeries(remove, 'foreground.revealed'), remove.eventIndex!, JOINT_ANALYSIS.stabilizeBand) } : null,
    teleport: Object.fromEntries(teleport.sets.map(item =>
      [item.name, stabilizeFrames(roiSeries(teleport, item.name), 0, JOINT_ANALYSIS.stabilizeBand)])),
  };
  // Same scene state (panel on, door open) seen from roomB and roomBAway.
  const l02 = { roomB: finalOf(panelOn, 'B.floor'), away: finalOf(away, 'B.floorAway'),
    change: relativeChange(finalOf(panelOn, 'B.floor'), finalOf(away, 'B.floorAway'), JOINT_ANALYSIS.maeEpsilon) };
  const panel = { wallGhost15: ghost(panelMove, 'panel.wall', panelStop, 15, 0), wallGhost30: ghost(panelMove, 'panel.wall', panelStop, 30, 0),
    belowPanel: responseTime(roiSeries(panelMove, 'B.backWallBelowPanel'), panelStop, { finalWindow: JOINT_ANALYSIS.finalWindow,
      baselineWindow: panelMove.eventIndex!, minimumChange: 1e-4 }) };
  const captured = run.phases.flatMap(phase => phase.frames.map(frame => frame.captured));
  const queries = run.phases.flatMap(phase => phase.frames.map(frame => frame.queries));
  const gpu = [...new Map(run.phases.flatMap(phase => phase.frames.flatMap(frame => frame.gpu ? [[frame.gpu.frame, frame.gpu.diffuseGiMs] as const] : []))).values()]
    .sort((a, b) => a - b);
  return { response, stability, t01, l02, panel,
    cost: { meanCapturedProbes: mean(captured), maxCapturedProbes: Math.max(...captured), meanQueriesPerUpdate: mean(queries),
      gpuDiffuseGiMs: gpu.length ? { samples: gpu.length, p50: gpu[Math.floor(gpu.length / 2)], p95: gpu[Math.floor(gpu.length * 0.95)] } : null } };
}

/** GI T90 = slowest non-negligible indirect ROI; settled (conservative) and first crossing. */
export function dynamicGates(name: string, metrics: ReturnType<typeof analyzeDynamics>, direct?: ReturnType<typeof analyzeDynamics>): Gate[] {
  const fps = JOINT_ANALYSIS.fps, gates: Gate[] = [];
  for (const [event, rois] of Object.entries(metrics.response)) {
    const relevant = Object.entries(rois).filter(([roi, value]) => !value.negligible && !roi.startsWith('A.sunPatch'));
    const settled = relevant.map(([, value]) => value.settledFrames);
    const worst = settled.some(value => value === null) ? null : relevant.length ? Math.max(...settled as number[]) : null;
    gates.push(gate('L03', `${name}.${event}.giT90SettledFrames@${fps}fps`, worst, '<=', 0.5 * fps,
      relevant.length ? undefined : 'no indirect ROI changed by more than the negligible threshold'));
  }
  if (direct) for (const event of ['sunOff', 'sunOn']) {
    gates.push(gate('L03', `direct.${event}.A.sunPatch.t90SettledFrames@${fps}fps`, direct.response[event]['A.sunPatch']?.settledFrames,
      '<=', 0.1 * fps));
  }
  for (const [roi, value] of Object.entries(metrics.stability)) {
    gates.push(gate('stability', `${name}.${roi}.fluctuation120`, value, '<=', 0.03, value === null ? 'near-dark ROI' : undefined));
  }
  gates.push(gate('L02', `${name}.B.floor.offscreenSteadyChange`, metrics.l02.change, '<=', 0.1));
  for (const [roi, frames] of Object.entries(metrics.t01.teleport)) gates.push(gate('T01', `${name}.teleport.${roi}.stabilizeFrames`, frames, '<=', fps));
  gates.push(gate('T01', `${name}.removeForeground.revealed.stabilizeFrames`, metrics.t01.revealed?.stabilizeFrames, '<=', fps));
  gates.push(gate('T01', `${name}.mover.patch.stabilizeFrames`, metrics.t01.moverPatchGhost.stabilizeFrames, '<=', fps));
  return gates;
}

function subtract(a: number[][], b: number[][]): number[][] { return a.map((value, i) => value.map((x, c) => x - b[i][c])); }
function meanRgb(values: number[][]): number[] { return [0, 1, 2].map(c => mean(values.map(value => value[c]))); }

export function analyzeSteady(run: Steady, label: string = run.update ?? 'unknown') {
  const rois: Record<string, unknown> = {};
  const gates: Gate[] = [];
  for (const name of Object.keys(run.multi)) {
    const indirect = subtract(run.multi[name], run.off[name]), single = subtract(run.single[name], run.off[name]);
    const entry: Record<string, unknown> = { indirect: meanRgb(indirect), single: meanRgb(single),
      multiBounceGain: luminance(meanRgb(indirect)) / Math.max(luminance(meanRgb(single)), 1e-6) };
    if (run.reference) {
      const total = run.reference.total[name], one = run.reference.oneBounce[name];
      const mae = normalizedMae(indirect, total, JOINT_ANALYSIS.maeEpsilon);
      const chroma = (rgb: number[]) => rgb[0] / Math.max(rgb[0] + rgb[1] + rgb[2], 1e-6);
      Object.assign(entry, { reference: meanRgb(total), referenceOneBounce: meanRgb(one), normalizedMae: mae,
        singleNormalizedMae: normalizedMae(single, one, JOINT_ANALYSIS.maeEpsilon),
        referenceMultiBounceGain: luminance(meanRgb(total)) / Math.max(luminance(meanRgb(one)), 1e-6),
        redChromaticity: chroma(meanRgb(indirect)), referenceRedChromaticity: chroma(meanRgb(total)) });
      gates.push(gate('L01', `${label}.${name}.indirectNormalizedMae`, mae, '<=', 0.15, 'initial target 10-15%'));
    }
    rois[name] = entry;
  }
  return { rois, gates };
}

/** Provisional L04 threshold (not frozen in benchmarks.md): interior luminance <= 1% of the sunlit ground. */
export function analyzeLeak(run: Leak, label: string = run.update ?? 'unknown') {
  const outside = luminance(run.on.outside[0]);
  const boxes: Record<string, unknown> = {};
  const gates: Gate[] = [];
  for (const name of ['blackBox', 'thinBox', 'slitBox'] as const) {
    const on = leakSummary(run.on[name], 0.01 * outside), off = leakSummary(run.off[name], 0.01 * outside);
    const indirect = subtract(run.on[name], run.off[name]);
    const entry: Record<string, unknown> = { on, off, indirectMeanLuminance: mean(indirect.map(luminance)),
      relativeToOutside: on.meanLuminance / outside };
    if (name !== 'slitBox') {
      gates.push(gate('L04', `${label}.${name}.interiorLuminance/outside`, on.meanLuminance / outside, '<=', 0.01,
        'provisional harness threshold; reference is exactly zero'));
    } else if (run.slitReference) {
      const mae = normalizedMae(indirect, run.slitReference, JOINT_ANALYSIS.maeEpsilon);
      Object.assign(entry, { referenceMeanLuminance: mean(run.slitReference.map(luminance)), indirectNormalizedMae: mae });
      gates.push(gate('L04', `${label}.slitBox.indirectNormalizedMae`, mae, '<=', 0.15, 'initial target 10-15%'));
    }
    boxes[name] = entry;
  }
  return { outsideLuminance: outside, boxes, gates };
}
