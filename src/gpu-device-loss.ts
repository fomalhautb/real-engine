type Listener = (info: GPUDeviceLostInfo) => void;
type Subscription = { listener?: Listener };
type LossState = { listeners: Set<Subscription>; info?: GPUDeviceLostInfo };
const devices = new WeakMap<GPUDevice, LossState>();

function deliver(subscription: Subscription, info: GPUDeviceLostInfo): void {
  const listener = subscription.listener;
  subscription.listener = undefined;
  // Owners report their own cleanup failures. One broken owner must not stop
  // another owner's release, or create an unhandled rejection on device.lost.
  try { listener?.(info); } catch { /* Isolate independently owned resources. */ }
}

/** One pending device.lost reaction per device, not per resource generation.
 * Dispose must unsubscribe so long-lived devices cannot retain retired owners.
 * Already-lost devices notify asynchronously; cancellation remains effective. */
export function subscribeGpuDeviceLoss(device: GPUDevice, listener: Listener): () => void {
  let state = devices.get(device);
  if (!state) {
    state = { listeners: new Set() }; devices.set(device, state);
    const observed = state;
    void device.lost.then(info => {
      observed.info = info;
      const pending = [...observed.listeners]; observed.listeners.clear();
      for (const subscription of pending) deliver(subscription, info);
    });
  }
  const subscription: Subscription = { listener };
  if (state.info) {
    const info = state.info;
    queueMicrotask(() => deliver(subscription, info));
  } else state.listeners.add(subscription);
  return () => {
    subscription.listener = undefined;
    state.listeners.delete(subscription);
  };
}

/** An owner of one GPUDevice, such as a Renderer. dispose() may destroy the device on purpose,
 * so the supervisor always unsubscribes before disposing an owner. */
export interface DeviceLossOwner {
  onDeviceLost(listener: Listener): () => void;
  dispose(): void;
}

/** Bounds for app-level recovery. Every value is validated; nothing retries without limit. */
export interface DeviceRecoveryPolicy {
  /** Factory attempts per observed loss, 1–8 (default 3). */
  maxAttempts?: number;
  /** Delay before attempt n uses entry n-1; the last entry repeats. Each 0–60,000 ms (default 0, 250, 1000). */
  backoffMs?: readonly number[];
  /** One factory attempt may take 100–120,000 ms (default 20,000). A late owner is disposed on arrival. */
  attemptTimeoutMs?: number;
  /** Losses accepted inside windowMs, 1–16 (default 3); one more stops recovery as a diagnosable failure. */
  maxLossesPerWindow?: number;
  /** Sliding loss window, 1,000–3,600,000 ms (default 60,000). */
  windowMs?: number;
}
export type ResolvedDeviceRecoveryPolicy = Readonly<Required<Omit<DeviceRecoveryPolicy, 'backoffMs'>> & { backoffMs: readonly number[] }>;
export type DeviceRecoveryState = 'idle' | 'starting' | 'running' | 'recovering' | 'failed' | 'disposed';
export interface DeviceRecoveryEvent {
  kind: 'started' | 'lost' | 'attempt' | 'attempt-failed' | 'recovered' | 'failed' | 'hook-error' | 'dispose-error';
  /** Supervisor clock in milliseconds. */
  atMs: number;
  /** Loss ordinal this event belongs to; 0 for the initial start. */
  loss: number;
  attempt?: number;
  reason?: string;
  message?: string;
  error?: string;
  durationMs?: number;
}
export interface DeviceRecoveryHooks<T> {
  /** Synchronous, before the lost owner is disposed: stop scheduling work against it. */
  lost?(info: GPUDeviceLostInfo, owner: T): void;
  /** The new owner is already subscribed for its own loss. It carries no GPU history. */
  recovered?(owner: T, report: { loss: number; attempts: number; durationMs: number }): void;
  /** Recovery stopped; the error message is the diagnosis. */
  failed?(error: Error): void;
}
export interface DeviceRecoveryDiagnostics {
  state: DeviceRecoveryState;
  losses: number;
  recoveries: number;
  failure: string | null;
  policy: ResolvedDeviceRecoveryPolicy;
  /** Most recent events, at most 64. */
  events: DeviceRecoveryEvent[];
}

const MAX_EVENTS = 64;
function bounded(value: number, low: number, high: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < low || value > high) throw new RangeError(`DEVICE_RECOVERY_POLICY: ${label} must be an integer in ${low}..${high}`);
  return value;
}
export function resolveDeviceRecoveryPolicy(policy: DeviceRecoveryPolicy = {}): ResolvedDeviceRecoveryPolicy {
  const backoffMs = [...(policy.backoffMs ?? [0, 250, 1000])];
  if (backoffMs.length < 1 || backoffMs.length > 8) throw new RangeError('DEVICE_RECOVERY_POLICY: backoffMs needs 1..8 entries');
  backoffMs.forEach((delay, index) => bounded(delay, 0, 60_000, `backoffMs[${index}]`));
  return Object.freeze({
    maxAttempts: bounded(policy.maxAttempts ?? 3, 1, 8, 'maxAttempts'),
    backoffMs: Object.freeze(backoffMs),
    attemptTimeoutMs: bounded(policy.attemptTimeoutMs ?? 20_000, 100, 120_000, 'attemptTimeoutMs'),
    maxLossesPerWindow: bounded(policy.maxLossesPerWindow ?? 3, 1, 16, 'maxLossesPerWindow'),
    windowMs: bounded(policy.windowMs ?? 60_000, 1000, 3_600_000, 'windowMs'),
  });
}
const message = (error: unknown): string => error instanceof Error ? error.message : String(error);

/** Explicit app-level device-loss recovery: observe loss → dispose the lost owner → ask the factory
 * for a new owner (which must request a new adapter and device and rebuild from retained CPU-side
 * inputs) → resume. A lost device is never reused. Attempts, delays, per-attempt time and loss
 * frequency are bounded; exhausting them leaves a failed state with a diagnosable message. */
export class GpuDeviceRecovery<T extends DeviceLossOwner> {
  readonly policy: ResolvedDeviceRecoveryPolicy;
  private owner?: T;
  private unsubscribe?: () => void;
  private status: DeviceRecoveryState = 'idle';
  private generation = 0;
  private losses = 0;
  private recoveries = 0;
  private failure?: Error;
  private readonly lossTimes: number[] = [];
  private readonly events: DeviceRecoveryEvent[] = [];

  constructor(
    private readonly factory: (context: { loss: number; attempt: number }) => Promise<T>,
    private readonly hooks: DeviceRecoveryHooks<T> = {},
    policy: DeviceRecoveryPolicy = {},
    private readonly now: () => number = () => performance.now(),
  ) { this.policy = resolveDeviceRecoveryPolicy(policy); }

  get state(): DeviceRecoveryState { return this.status; }
  /** The live owner; undefined while recovering, after failure, or after disposal. */
  get current(): T | undefined { return this.status === 'running' ? this.owner : undefined; }

  /** Initial creation is a single attempt: unavailable WebGPU or a bad scene is not a device loss. */
  async start(): Promise<T> {
    if (this.status !== 'idle') throw new Error('DEVICE_RECOVERY_STATE: start() requires an idle supervisor, not ' + this.status);
    this.status = 'starting';
    const generation = this.generation;
    let owner: T;
    try { owner = await this.factory({ loss: 0, attempt: 1 }); } catch (error) {
      if (generation === this.generation) { this.failure = error instanceof Error ? error : new Error(message(error)); this.status = 'failed'; }
      throw error;
    }
    if (generation !== this.generation) { this.release(owner); throw new Error('DEVICE_RECOVERY_DISPOSED'); }
    this.adopt(owner);
    this.log({ kind: 'started', loss: 0, attempt: 1 });
    return owner;
  }

  diagnostics(): DeviceRecoveryDiagnostics {
    return { state: this.status, losses: this.losses, recoveries: this.recoveries,
      failure: this.failure?.message ?? null, policy: this.policy, events: this.events.map(event => ({ ...event })) };
  }

  /** Cancels any in-flight recovery and releases the live owner. Idempotent. */
  dispose(): void {
    if (this.status === 'disposed') return;
    this.generation++;
    this.status = 'disposed';
    this.unsubscribe?.(); this.unsubscribe = undefined;
    const owner = this.owner; this.owner = undefined;
    if (owner) this.release(owner);
  }

  private log(event: Omit<DeviceRecoveryEvent, 'atMs'>): void {
    this.events.push({ atMs: this.now(), ...event });
    if (this.events.length > MAX_EVENTS) this.events.splice(0, this.events.length - MAX_EVENTS);
  }

  private release(owner: T): void {
    try { owner.dispose(); } catch (error) { this.log({ kind: 'dispose-error', loss: this.losses, error: message(error) }); }
  }

  private adopt(owner: T): void {
    const generation = this.generation;
    this.owner = owner; this.status = 'running';
    this.unsubscribe = owner.onDeviceLost(info => {
      if (generation === this.generation && this.owner === owner && this.status === 'running') void this.recover(owner, info);
    });
  }

  private fail(error: Error): void {
    this.failure = error; this.status = 'failed';
    this.log({ kind: 'failed', loss: this.losses, error: error.message });
    try { this.hooks.failed?.(error); } catch (hookError) { this.log({ kind: 'hook-error', loss: this.losses, error: message(hookError) }); }
  }

  private sleep(ms: number): Promise<void> { return new Promise(resolve => setTimeout(resolve, ms)); }

  /** Resolves the factory result, or rejects after attemptTimeoutMs; a late owner is released. */
  private attempt(loss: number, attempt: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        reject(new Error(`DEVICE_RECOVERY_TIMEOUT: attempt ${attempt} exceeded ${this.policy.attemptTimeoutMs} ms`));
      }, this.policy.attemptTimeoutMs);
      let pending: Promise<T>;
      try { pending = this.factory({ loss, attempt }); } catch (error) { pending = Promise.reject(error); }
      pending.then(owner => {
        if (settled) { this.release(owner); return; }
        settled = true; clearTimeout(timer); resolve(owner);
      }, error => {
        if (settled) return;
        settled = true; clearTimeout(timer); reject(error);
      });
    });
  }

  private async recover(lost: T, info: GPUDeviceLostInfo): Promise<void> {
    const generation = this.generation, started = this.now();
    const loss = ++this.losses;
    this.status = 'recovering'; this.unsubscribe = undefined;
    this.log({ kind: 'lost', loss, reason: info.reason, message: info.message });
    try { this.hooks.lost?.(info, lost); } catch (error) { this.log({ kind: 'hook-error', loss, error: message(error) }); }
    // Every other listener on the lost device runs before its owner is released.
    await this.sleep(0);
    if (generation !== this.generation) return;
    this.owner = undefined;
    this.release(lost);
    while (this.lossTimes.length && started - this.lossTimes[0] >= this.policy.windowMs) this.lossTimes.shift();
    this.lossTimes.push(started);
    const detail = `${info.reason || 'unknown'} device loss (${info.message || 'no message'})`;
    if (this.lossTimes.length > this.policy.maxLossesPerWindow) {
      this.fail(new Error(`DEVICE_RECOVERY_LIMIT: ${this.lossTimes.length} device losses within ${this.policy.windowMs} ms `
        + `exceed ${this.policy.maxLossesPerWindow}; last: ${detail}`));
      return;
    }
    let lastError = 'none';
    for (let attempt = 1; attempt <= this.policy.maxAttempts; attempt++) {
      const delay = this.policy.backoffMs[Math.min(attempt - 1, this.policy.backoffMs.length - 1)];
      if (delay > 0) await this.sleep(delay);
      if (generation !== this.generation) return;
      this.log({ kind: 'attempt', loss, attempt });
      const attemptStarted = this.now();
      let owner: T;
      try { owner = await this.attempt(loss, attempt); } catch (error) {
        lastError = message(error);
        this.log({ kind: 'attempt-failed', loss, attempt, error: lastError, durationMs: this.now() - attemptStarted });
        continue;
      }
      if (generation !== this.generation) { this.release(owner); return; }
      this.adopt(owner); this.recoveries++;
      const durationMs = this.now() - started;
      this.log({ kind: 'recovered', loss, attempt, durationMs });
      try { this.hooks.recovered?.(owner, { loss, attempts: attempt, durationMs }); } catch (error) { this.log({ kind: 'hook-error', loss, error: message(error) }); }
      return;
    }
    this.fail(new Error(`DEVICE_RECOVERY_FAILED: ${this.policy.maxAttempts} attempts after ${detail}; last error: ${lastError}`));
  }
}
