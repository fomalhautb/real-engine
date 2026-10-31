/** Change-driven, bounded and staleness-limited capture schedule for the fixed
 * world-space diffuse probe grid. The schedule is planned on the CPU from
 * explicit scene change sets; it never reads GPU probe data back.
 *
 * Guarantee: every scheduled update captures the `staleReserve =
 * ceil(probeCount / maxStalenessUpdates)` least recently captured probes first,
 * so no probe goes more than `maxStalenessUpdates` completed GI updates without
 * a capture. Dirty probes then compete for the remaining budget: first captures
 * after a change before convergence recaptures, each by estimated influence
 * (solid angle) and view relevance. A hard reset recaptures the whole grid once. */
import type { Bounds3 } from './deformation';

type Vec3 = readonly [number, number, number];
export type DiffuseGiUpdateMode = 'full' | 'scheduled';
/** Why a probe was marked dirty. 'light' and 'external' without a region are global. */
export type GiChangeReason = 'transform' | 'instance' | 'material' | 'topology' | 'light' | 'external';

export interface DiffuseGiUpdateOptions {
  /** 'full' (default) recaptures every probe each update, as before. 'scheduled'
   * recaptures a bounded, prioritized subset with an explicit staleness bound. */
  mode?: DiffuseGiUpdateMode;
  /** Probes captured by one scheduled update, 1..probeCount. */
  maxProbesPerUpdate?: number;
  /** Largest number of completed GI updates between two captures of one probe. */
  maxStalenessUpdates?: number;
  /** Prioritized recaptures after a change (multi-bounce convergence), 1..16. */
  convergenceUpdates?: number;
  /** Smallest estimated solid-angle fraction of a local change that marks a probe. */
  dirtyThreshold?: number;
  /** Priority multiplier (0, 1] for dirty probes outside the expanded view frustum. */
  offscreenPriority?: number;
}

export interface ResolvedDiffuseGiUpdateOptions {
  readonly mode: DiffuseGiUpdateMode;
  readonly maxProbesPerUpdate: number;
  readonly maxStalenessUpdates: number;
  readonly convergenceUpdates: number;
  readonly dirtyThreshold: number;
  readonly offscreenPriority: number;
  /** Least recently captured probes taken first by every scheduled update. */
  readonly staleReserve: number;
}

const DEFAULT_STALENESS = 8;
const DEFAULT_CONVERGENCE = 3;
const DEFAULT_DIRTY_THRESHOLD = 0.002;
const DEFAULT_OFFSCREEN_PRIORITY = 0.25;
/** Region marks beyond this per update are merged into one conservative box. */
export const GI_MAX_CHANGE_REGIONS_PER_UPDATE = 64;

export function resolveDiffuseGiUpdateOptions(options: DiffuseGiUpdateOptions | undefined,
  probeCount: number): ResolvedDiffuseGiUpdateOptions {
  if (!Number.isSafeInteger(probeCount) || probeCount < 1) throw new RangeError('DIFFUSE_GI_UPDATE_PROBES');
  if (options !== undefined && (options === null || typeof options !== 'object')) throw new RangeError('DIFFUSE_GI_UPDATE_OPTIONS');
  const mode = options?.mode ?? 'full';
  if (mode !== 'full' && mode !== 'scheduled') throw new RangeError('DIFFUSE_GI_UPDATE_MODE');
  const integer = (value: number | undefined, fallback: number, min: number, max: number, label: string) => {
    const resolved = value ?? fallback;
    if (!Number.isSafeInteger(resolved) || resolved < min || resolved > max) throw new RangeError('DIFFUSE_GI_UPDATE_' + label);
    return resolved;
  };
  const number = (value: number | undefined, fallback: number, label: string) => {
    const resolved = value ?? fallback;
    if (typeof resolved !== 'number' || !Number.isFinite(resolved) || resolved <= 0 || resolved > 1) {
      throw new RangeError('DIFFUSE_GI_UPDATE_' + label);
    }
    return resolved;
  };
  if (mode === 'full') {
    const full = Object.freeze({ mode, maxProbesPerUpdate: probeCount, maxStalenessUpdates: 1, convergenceUpdates: 1,
      dirtyThreshold: DEFAULT_DIRTY_THRESHOLD, offscreenPriority: 1, staleReserve: probeCount });
    // Scheduling parameters have no meaning here; only an already resolved
    // full policy (identical values) may be resolved again.
    for (const key of ['maxProbesPerUpdate', 'maxStalenessUpdates', 'convergenceUpdates', 'dirtyThreshold', 'offscreenPriority'] as const) {
      if (options?.[key] !== undefined && options[key] !== full[key]) throw new RangeError('DIFFUSE_GI_UPDATE_FULL_MODE_OPTION');
    }
    return full;
  }
  const maxStalenessUpdates = integer(options?.maxStalenessUpdates, DEFAULT_STALENESS, 1, 1024, 'STALENESS');
  const staleReserve = Math.ceil(probeCount / maxStalenessUpdates);
  const maxProbesPerUpdate = integer(options?.maxProbesPerUpdate,
    Math.min(probeCount, Math.max(staleReserve, Math.ceil(probeCount / 4))), 1, probeCount, 'BUDGET');
  if (maxProbesPerUpdate < staleReserve) throw new RangeError('DIFFUSE_GI_UPDATE_STALENESS_BUDGET');
  return Object.freeze({ mode, maxProbesPerUpdate, maxStalenessUpdates, staleReserve,
    convergenceUpdates: integer(options?.convergenceUpdates, DEFAULT_CONVERGENCE, 1, 16, 'CONVERGENCE'),
    dirtyThreshold: number(options?.dirtyThreshold, DEFAULT_DIRTY_THRESHOLD, 'DIRTY_THRESHOLD'),
    offscreenPriority: number(options?.offscreenPriority, DEFAULT_OFFSCREEN_PRIORITY, 'OFFSCREEN_PRIORITY') });
}

export interface GiProbeGrid {
  readonly origin: Vec3;
  readonly spacing: Vec3;
  readonly counts: Vec3;
  /** Capture ray length; changes farther than this cannot be seen directly. */
  readonly maxDistance: number;
}

export interface GiScheduleView {
  readonly eye: Vec3;
  /** Column-major WebGPU view-projection, depth in [0, 1]. */
  readonly viewProjection?: ArrayLike<number>;
}

export interface GiSchedulePlan {
  readonly frame: number;
  /** Probe indices to capture, in slot order; length is `count`. */
  readonly probes: Uint32Array;
  readonly count: number;
  readonly fullRefresh: boolean;
  readonly fullRefreshReason: string | null;
  readonly staleProbes: number;
  readonly dirtyProbes: number;
}

export interface GiScheduleDiagnostics {
  readonly mode: DiffuseGiUpdateMode;
  readonly probeCount: number;
  readonly maxProbesPerUpdate: number;
  readonly maxStalenessUpdates: number;
  readonly staleReserve: number;
  readonly convergenceUpdates: number;
  readonly dirtyThreshold: number;
  readonly offscreenPriority: number;
  /** Version of global light dependencies; incremented by global marks and resets. */
  readonly dependencyVersion: number;
  readonly lastCommittedFrame: number;
  readonly lastPlan: Omit<GiSchedulePlan, 'probes'> | null;
  /** Updates since each probe's capture, measured at the last committed update. */
  readonly maxAgeUpdates: number;
  readonly meanAgeUpdates: number;
  readonly neverCapturedProbes: number;
  /** Probes with prioritized recaptures still pending, and those recaptures. */
  readonly pendingDirtyProbes: number;
  readonly pendingCaptures: number;
  /** Dirty probes not captured since their latest change (pre-change data). */
  readonly unrefreshedDirtyProbes: number;
  readonly oldestPendingDirtyUpdates: number;
  /** Probes captured before the latest global dependency version. */
  readonly staleDependencyProbes: number;
  readonly totals: { readonly updates: number; readonly captures: number; readonly fullRefreshes: number;
    readonly regionMarks: number; readonly mergedRegionMarks: number; readonly globalMarks: number;
    readonly markedProbes: number };
}

/** Planner for one fixed grid. Not thread-safe; plan → commit once per update. */
export class GiProbeScheduler {
  readonly probeCount: number;
  private options: ResolvedDiffuseGiUpdateOptions;
  private readonly positions: Float64Array;
  private readonly cellRadius: number;
  /** Refresh order key: smaller is less recently captured. */
  private readonly order: Float64Array;
  private readonly lastCapture: Uint32Array;
  private readonly captured: Uint8Array;
  private readonly capturedVersion: Uint32Array;
  private readonly dirtyWeight: Float32Array;
  private readonly pending: Uint8Array;
  /** Marked since the last capture: still shows data from before the change. */
  private readonly unrefreshed: Uint8Array;
  private readonly dirtySince: Uint32Array;
  private readonly selected: Uint8Array;
  private readonly inView: Uint8Array;
  private readonly byAge: number[];
  private readonly output: Uint32Array;
  private regions: { region: Bounds3; reason: GiChangeReason }[] = [];
  private fullRefreshReason: string | null = 'initial';
  private dependencyVersion = 0;
  private committedFrame = 0;
  private view?: GiScheduleView;
  private plannedFrame = 0;
  private lastPlan: GiSchedulePlan | null = null;
  private lastCommitted: Omit<GiSchedulePlan, 'probes'> | null = null;
  private readonly totals = { updates: 0, captures: 0, fullRefreshes: 0, regionMarks: 0, mergedRegionMarks: 0,
    globalMarks: 0, markedProbes: 0 };

  constructor(private readonly grid: GiProbeGrid, options: ResolvedDiffuseGiUpdateOptions) {
    const [x, y, z] = grid.counts;
    this.probeCount = x * y * z;
    if (!Number.isSafeInteger(this.probeCount) || this.probeCount < 1 || this.probeCount > 65_536) {
      throw new RangeError('DIFFUSE_GI_SCHEDULE_GRID');
    }
    if (options.maxProbesPerUpdate > this.probeCount) throw new RangeError('DIFFUSE_GI_UPDATE_BUDGET');
    this.options = options;
    const n = this.probeCount;
    this.positions = new Float64Array(n * 3);
    for (let index = 0; index < n; index++) {
      const cx = index % x, cy = Math.floor(index / x) % y, cz = Math.floor(index / (x * y));
      this.positions.set([grid.origin[0] + cx * grid.spacing[0], grid.origin[1] + cy * grid.spacing[1],
        grid.origin[2] + cz * grid.spacing[2]], index * 3);
    }
    this.cellRadius = 0.5 * Math.hypot(...grid.spacing);
    this.order = new Float64Array(n).map((_, index) => index - n);
    this.lastCapture = new Uint32Array(n);
    this.captured = new Uint8Array(n);
    this.capturedVersion = new Uint32Array(n);
    this.dirtyWeight = new Float32Array(n);
    this.pending = new Uint8Array(n);
    this.unrefreshed = new Uint8Array(n);
    this.dirtySince = new Uint32Array(n);
    this.selected = new Uint8Array(n);
    this.inView = new Uint8Array(n);
    this.byAge = Array.from({ length: n }, (_, index) => index);
    this.output = new Uint32Array(n);
  }

  get mode(): DiffuseGiUpdateMode { return this.options.mode; }
  get resolvedOptions(): ResolvedDiffuseGiUpdateOptions { return this.options; }

  /** Replace the policy; the caller must request a full refresh for the switch. */
  setOptions(options: ResolvedDiffuseGiUpdateOptions): void {
    if (options.maxProbesPerUpdate > this.probeCount) throw new RangeError('DIFFUSE_GI_UPDATE_BUDGET');
    this.options = options;
  }

  /** Hard reset: the next plan recaptures every probe once, then converges at budget. */
  requestFullRefresh(reason: string): void {
    this.fullRefreshReason ??= reason;
    this.dependencyVersion = (this.dependencyVersion + 1) >>> 0;
  }

  /** Global light/sky dependency changed; every probe becomes dirty, none is rejected. */
  markAll(reason: GiChangeReason, frame: number): void {
    this.totals.globalMarks++;
    this.dependencyVersion = (this.dependencyVersion + 1) >>> 0;
    for (let probe = 0; probe < this.probeCount; probe++) this.mark(probe, 1, frame);
    this.totals.markedProbes += this.probeCount;
  }

  /** Queue a local change; regions are applied when the next update is planned. */
  markRegion(region: Bounds3, reason: GiChangeReason): void {
    if (!region || region.min.length !== 3 || region.max.length !== 3
      || ![...region.min, ...region.max].every(Number.isFinite)
      || region.min.some((value, axis) => value > region.max[axis])) throw new RangeError('DIFFUSE_GI_CHANGE_REGION');
    this.regions.push({ region: { min: [...region.min], max: [...region.max] }, reason });
  }

  setView(view: GiScheduleView | undefined): void { this.view = view; }

  plan(frame: number): GiSchedulePlan {
    if (!Number.isSafeInteger(frame) || frame <= this.committedFrame || frame > 0xffffffff) throw new RangeError('DIFFUSE_GI_SCHEDULE_FRAME');
    this.applyRegions(frame);
    const n = this.probeCount, output = this.output;
    let count = 0, stale = 0, dirty = 0;
    const full = this.fullRefreshReason !== null || this.options.maxProbesPerUpdate >= n;
    if (full) {
      for (let probe = 0; probe < n; probe++) output[probe] = probe;
      count = n;
    } else {
      this.selected.fill(0);
      const order = this.order;
      this.byAge.sort((a, b) => order[a] - order[b]);
      for (; count < this.options.staleReserve; count++) {
        const probe = this.byAge[count];
        output[count] = probe; this.selected[probe] = 1; stale++;
      }
      const remaining = this.options.maxProbesPerUpdate - count;
      if (remaining > 0) {
        this.classifyView();
        const candidates: number[] = [];
        for (let probe = 0; probe < n; probe++) if (this.pending[probe] && !this.selected[probe]) candidates.push(probe);
        const score = (probe: number) => this.dirtyWeight[probe] * (this.inView[probe] ? 1 : this.options.offscreenPriority);
        // First captures after a change outrank feedback-convergence recaptures.
        const unrefreshed = this.unrefreshed;
        candidates.sort((a, b) => unrefreshed[b] - unrefreshed[a] || score(b) - score(a) || order[a] - order[b]);
        for (let i = 0; i < candidates.length && i < remaining; i++) {
          output[count++] = candidates[i]; this.selected[candidates[i]] = 1; dirty++;
        }
      }
    }
    this.plannedFrame = frame;
    this.lastPlan = Object.freeze({ frame, probes: output.subarray(0, count), count, fullRefresh: full,
      fullRefreshReason: full ? this.fullRefreshReason ?? 'budget-covers-grid' : null, staleProbes: stale, dirtyProbes: dirty });
    return this.lastPlan;
  }

  /** Record that the planned captures were encoded for the GPU. */
  commit(plan: GiSchedulePlan): void {
    if (plan !== this.lastPlan || plan.frame !== this.plannedFrame) throw new Error('DIFFUSE_GI_SCHEDULE_PLAN_MISMATCH');
    const n = this.probeCount;
    if (plan.fullRefresh && this.fullRefreshReason !== null) {
      // The grid is complete again. Every probe still needs feedback bounces,
      // so converge the whole field at the bounded budget afterwards.
      this.fullRefreshReason = null;
      this.totals.fullRefreshes++;
      for (let probe = 0; probe < n; probe++) {
        this.pending[probe] = 0; this.dirtyWeight[probe] = 0; this.unrefreshed[probe] = 0;
        this.mark(probe, 1, plan.frame);
      }
    }
    for (let slot = 0; slot < plan.count; slot++) {
      const probe = plan.probes[slot];
      this.lastCapture[probe] = plan.frame; this.captured[probe] = 1;
      this.capturedVersion[probe] = this.dependencyVersion;
      this.order[probe] = plan.frame * n + slot;
      this.unrefreshed[probe] = 0;
      if (this.pending[probe]) {
        this.pending[probe]--;
        // Fresh changes outrank probes that are only converging feedback.
        this.dirtyWeight[probe] = this.pending[probe] ? this.dirtyWeight[probe] * 0.5 : 0;
      }
    }
    this.committedFrame = plan.frame;
    this.totals.updates++; this.totals.captures += plan.count;
    this.lastCommitted = { frame: plan.frame, count: plan.count, fullRefresh: plan.fullRefresh,
      fullRefreshReason: plan.fullRefreshReason, staleProbes: plan.staleProbes, dirtyProbes: plan.dirtyProbes };
    this.lastPlan = null;
  }

  /** Age in committed updates; probes never captured report Infinity. */
  ageOf(probe: number, frame = this.committedFrame): number {
    return this.captured[probe] ? frame - this.lastCapture[probe] : Infinity;
  }

  diagnostics(): GiScheduleDiagnostics {
    let maxAge = 0, sumAge = 0, never = 0, pendingProbes = 0, pendingCaptures = 0, oldestDirty = 0, staleVersion = 0;
    let unrefreshed = 0;
    for (let probe = 0; probe < this.probeCount; probe++) {
      if (!this.captured[probe]) { never++; continue; }
      const age = this.committedFrame - this.lastCapture[probe];
      maxAge = Math.max(maxAge, age); sumAge += age;
      if (this.capturedVersion[probe] !== this.dependencyVersion) staleVersion++;
      if (this.pending[probe]) {
        pendingProbes++; pendingCaptures += this.pending[probe];
        unrefreshed += this.unrefreshed[probe];
        oldestDirty = Math.max(oldestDirty, Math.max(0, this.committedFrame - this.dirtySince[probe]));
      }
    }
    const captured = this.probeCount - never;
    const plan = this.lastPlan ?? this.lastCommitted;
    return Object.freeze({ ...this.options, probeCount: this.probeCount, dependencyVersion: this.dependencyVersion,
      lastCommittedFrame: this.committedFrame,
      lastPlan: plan ? { frame: plan.frame, count: plan.count, fullRefresh: plan.fullRefresh,
        fullRefreshReason: plan.fullRefreshReason, staleProbes: plan.staleProbes, dirtyProbes: plan.dirtyProbes } : null,
      maxAgeUpdates: maxAge, meanAgeUpdates: captured ? sumAge / captured : 0, neverCapturedProbes: never,
      pendingDirtyProbes: pendingProbes, pendingCaptures, unrefreshedDirtyProbes: unrefreshed,
      oldestPendingDirtyUpdates: oldestDirty,
      staleDependencyProbes: staleVersion, totals: { ...this.totals } });
  }

  private mark(probe: number, weight: number, frame: number): void {
    if (!this.pending[probe]) this.dirtySince[probe] = frame;
    this.pending[probe] = Math.max(this.pending[probe], this.options.convergenceUpdates);
    this.dirtyWeight[probe] = Math.max(this.dirtyWeight[probe], weight);
    this.unrefreshed[probe] = 1;
  }

  /** Solid-angle fraction of the region's bounding sphere seen from each probe. */
  private applyRegions(frame: number): void {
    if (!this.regions.length) return;
    let regions = this.regions;
    this.regions = [];
    if (regions.length > GI_MAX_CHANGE_REGIONS_PER_UPDATE) {
      // Bounded CPU work: keep the first marks exact, merge the rest conservatively.
      const merged = { min: [...regions[GI_MAX_CHANGE_REGIONS_PER_UPDATE - 1].region.min] as [number, number, number],
        max: [...regions[GI_MAX_CHANGE_REGIONS_PER_UPDATE - 1].region.max] as [number, number, number] };
      for (const { region } of regions.slice(GI_MAX_CHANGE_REGIONS_PER_UPDATE)) for (let axis = 0; axis < 3; axis++) {
        merged.min[axis] = Math.min(merged.min[axis], region.min[axis]);
        merged.max[axis] = Math.max(merged.max[axis], region.max[axis]);
      }
      this.totals.mergedRegionMarks += regions.length - GI_MAX_CHANGE_REGIONS_PER_UPDATE + 1;
      regions = [...regions.slice(0, GI_MAX_CHANGE_REGIONS_PER_UPDATE - 1), { region: merged, reason: 'external' }];
    }
    const threshold = this.options.dirtyThreshold, reach = this.grid.maxDistance;
    for (const { region } of regions) {
      this.totals.regionMarks++;
      const center = [0, 1, 2].map(axis => 0.5 * (region.min[axis] + region.max[axis]));
      const radius = 0.5 * Math.hypot(...[0, 1, 2].map(axis => region.max[axis] - region.min[axis])) + 1e-6;
      for (let probe = 0; probe < this.probeCount; probe++) {
        const p = probe * 3;
        const distance = Math.hypot(this.positions[p] - center[0], this.positions[p + 1] - center[1], this.positions[p + 2] - center[2]);
        if (distance - radius > reach + this.cellRadius) continue;
        // Priority is the fraction of the probe's directions that see the
        // change (1 inside its bounding sphere): larger means a larger change.
        const fraction = distance <= radius ? 1 : 0.5 * (1 - Math.sqrt(Math.max(0, 1 - (radius / distance) ** 2)));
        // Conservative marking: a probe within one cell of the change always
        // recaptures, whatever its solid angle.
        if (fraction < threshold && distance > radius + this.cellRadius) continue;
        this.mark(probe, fraction, frame);
        this.totals.markedProbes++;
      }
    }
  }

  /** Probe spheres (cell half-diagonal) against the six view-frustum planes. */
  private classifyView(): void {
    const vp = this.view?.viewProjection;
    if (!vp || vp.length !== 16) { this.inView.fill(1); return; }
    const planes: number[][] = [];
    const row = (i: number) => [vp[i], vp[4 + i], vp[8 + i], vp[12 + i]];
    const r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    for (const plane of [r3.map((v, i) => v + r0[i]), r3.map((v, i) => v - r0[i]), r3.map((v, i) => v + r1[i]),
      r3.map((v, i) => v - r1[i]), r2, r3.map((v, i) => v - r2[i])]) {
      const length = Math.hypot(plane[0], plane[1], plane[2]);
      if (!(length > 0) || !Number.isFinite(length)) { this.inView.fill(1); return; }
      planes.push(plane.map(value => value / length));
    }
    for (let probe = 0; probe < this.probeCount; probe++) {
      const p = probe * 3;
      let inside = 1;
      for (const plane of planes) {
        if (plane[0] * this.positions[p] + plane[1] * this.positions[p + 1] + plane[2] * this.positions[p + 2] + plane[3]
          < -this.cellRadius) { inside = 0; break; }
      }
      this.inView[probe] = inside;
    }
  }
}
