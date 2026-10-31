import { describe, expect, it, vi } from 'vitest';
import { GpuDeviceRecovery, resolveDeviceRecoveryPolicy, subscribeGpuDeviceLoss } from '../src/gpu-device-loss';

function device() {
  let lose!: (info: GPUDeviceLostInfo) => void;
  const lost = new Promise<GPUDeviceLostInfo>(resolve => { lose = resolve; });
  const then = vi.spyOn(lost, 'then');
  return { gpu: { lost } as GPUDevice, lose, then };
}
const loss = { reason: 'destroyed', message: 'test device loss' } as GPUDeviceLostInfo;

describe('shared GPU device loss subscriptions', () => {
  it('uses one promise reaction across repeated owner generations and releases every cancelled listener', async () => {
    const d = device(), retired = vi.fn(), live = vi.fn();
    for (let generation = 0; generation < 1000; generation++) {
      const unsubscribe = subscribeGpuDeviceLoss(d.gpu, retired); unsubscribe(); unsubscribe();
    }
    subscribeGpuDeviceLoss(d.gpu, live);
    expect(d.then).toHaveBeenCalledOnce();
    d.lose(loss); await Promise.resolve();
    expect(retired).not.toHaveBeenCalled(); expect(live).toHaveBeenCalledExactlyOnceWith(loss);
  });

  it('isolates callback failure and allows cancellation during loss delivery', async () => {
    const d = device(), later = vi.fn(), last = vi.fn();
    let unsubscribeLater = () => {};
    subscribeGpuDeviceLoss(d.gpu, () => { unsubscribeLater(); throw new Error('OWNER_CLEANUP_FAILURE'); });
    unsubscribeLater = subscribeGpuDeviceLoss(d.gpu, later);
    subscribeGpuDeviceLoss(d.gpu, last);
    d.lose(loss); await Promise.resolve();
    expect(later).not.toHaveBeenCalled(); expect(last).toHaveBeenCalledExactlyOnceWith(loss);
  });

  it('notifies registrations after loss asynchronously and honors cancellation', async () => {
    const d = device(), first = vi.fn();
    subscribeGpuDeviceLoss(d.gpu, first); d.lose(loss); await Promise.resolve();
    const cancelled = vi.fn(), active = vi.fn();
    subscribeGpuDeviceLoss(d.gpu, cancelled)(); subscribeGpuDeviceLoss(d.gpu, active);
    expect(active).not.toHaveBeenCalled(); await Promise.resolve();
    expect(cancelled).not.toHaveBeenCalled(); expect(active).toHaveBeenCalledExactlyOnceWith(loss);
    expect(d.then).toHaveBeenCalledOnce();
  });

  it('handles first registration on an already-resolved device and keeps different devices independent', async () => {
    const a = device(), b = device(), cancelled = vi.fn(), active = vi.fn();
    a.lose(loss); subscribeGpuDeviceLoss(a.gpu, cancelled)(); subscribeGpuDeviceLoss(b.gpu, active);
    await Promise.resolve(); expect(cancelled).not.toHaveBeenCalled(); expect(active).not.toHaveBeenCalled();
    b.lose(loss); await Promise.resolve(); expect(active).toHaveBeenCalledExactlyOnceWith(loss);
    expect(a.then).toHaveBeenCalledOnce(); expect(b.then).toHaveBeenCalledOnce();
  });
});

type FakeOwner = { id: number; disposed: number; lose(info?: GPUDeviceLostInfo): void;
  onDeviceLost(listener: (info: GPUDeviceLostInfo) => void): () => void; dispose(): void };
function owners() {
  let next = 0;
  const created: FakeOwner[] = [];
  const make = (): FakeOwner => {
    let listener: ((info: GPUDeviceLostInfo) => void) | undefined;
    const owner: FakeOwner = { id: ++next, disposed: 0,
      lose(info = loss) { const active = listener; listener = undefined; active?.(info); },
      onDeviceLost(value) { listener = value; return () => { if (listener === value) listener = undefined; }; },
      dispose() { owner.disposed++; listener = undefined; } };
    created.push(owner);
    return owner;
  };
  return { created, make };
}
const flush = async (ms = 0) => { await vi.advanceTimersByTimeAsync(ms); };

describe('bounded device-loss recovery supervisor', () => {
  it('disposes the lost owner, rebuilds once and resumes with a newly subscribed owner', async () => {
    vi.useFakeTimers();
    try {
      const pool = owners(), lost = vi.fn(), recovered = vi.fn(), failed = vi.fn();
      const factory = vi.fn(async () => pool.make());
      const recovery = new GpuDeviceRecovery(factory, { lost, recovered, failed }, {}, () => Date.now());
      const first = await recovery.start();
      expect(recovery.state).toBe('running'); expect(recovery.current).toBe(first);
      first.lose();
      expect(recovery.state).toBe('recovering'); expect(recovery.current).toBeUndefined();
      expect(lost).toHaveBeenCalledExactlyOnceWith(loss, first);
      expect(first.disposed).toBe(0); // other loss listeners run before release
      await flush();
      expect(first.disposed).toBe(1);
      expect(factory).toHaveBeenCalledTimes(2);
      expect(factory).toHaveBeenLastCalledWith({ loss: 1, attempt: 1 });
      const second = pool.created[1];
      expect(recovery.state).toBe('running'); expect(recovery.current).toBe(second);
      expect(recovered).toHaveBeenCalledExactlyOnceWith(second, expect.objectContaining({ loss: 1, attempts: 1 }));
      expect(failed).not.toHaveBeenCalled();
      expect(recovery.diagnostics().events.map(event => event.kind)).toEqual(['started', 'lost', 'attempt', 'recovered']);
      expect(recovery.diagnostics().events[1]).toEqual(expect.objectContaining({ reason: 'destroyed', message: 'test device loss' }));
      first.lose(); await flush(); // a retired owner cannot trigger another recovery
      expect(factory).toHaveBeenCalledTimes(2);
      recovery.dispose(); recovery.dispose();
      expect(second.disposed).toBe(1); expect(recovery.state).toBe('disposed');
      second.lose(); await flush(); expect(factory).toHaveBeenCalledTimes(2);
    } finally { vi.useRealTimers(); }
  });

  it('retries with bounded backoff and reports a diagnosable failure after the last attempt', async () => {
    vi.useFakeTimers();
    try {
      const pool = owners(), failed = vi.fn();
      let calls = 0;
      const recovery = new GpuDeviceRecovery(async () => {
        if (calls++ === 0) return pool.make();
        throw new Error('ADAPTER_UNAVAILABLE: no compatible WebGPU adapter.');
      }, { failed }, { maxAttempts: 3, backoffMs: [0, 100, 400] }, () => Date.now());
      (await recovery.start()).lose({ reason: 'unknown', message: 'GPU process exited' } as GPUDeviceLostInfo);
      await flush();
      expect(calls).toBe(2);
      await flush(99); expect(calls).toBe(2);
      await flush(1); expect(calls).toBe(3);
      await flush(399); expect(calls).toBe(3);
      await flush(1); expect(calls).toBe(4);
      await flush(10_000); expect(calls).toBe(4);
      expect(recovery.state).toBe('failed');
      const diagnostics = recovery.diagnostics();
      expect(diagnostics.failure).toBe('DEVICE_RECOVERY_FAILED: 3 attempts after unknown device loss (GPU process exited); '
        + 'last error: ADAPTER_UNAVAILABLE: no compatible WebGPU adapter.');
      expect(failed).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ message: diagnostics.failure }));
      expect(diagnostics.events.filter(event => event.kind === 'attempt-failed').map(event => event.attempt)).toEqual([1, 2, 3]);
      expect(pool.created[0].disposed).toBe(1);
    } finally { vi.useRealTimers(); }
  });

  it('recovers on a later attempt, times out a hung attempt and releases its late owner', async () => {
    vi.useFakeTimers();
    try {
      const pool = owners();
      let calls = 0, releaseHung!: () => void;
      const recovery = new GpuDeviceRecovery(() => {
        calls++;
        if (calls === 2) return new Promise<FakeOwner>(resolve => { releaseHung = () => resolve(pool.make()); });
        return Promise.resolve(pool.make());
      }, {}, { attemptTimeoutMs: 500, backoffMs: [0] }, () => Date.now());
      (await recovery.start()).lose();
      await flush(); expect(calls).toBe(2); expect(recovery.state).toBe('recovering');
      await flush(500); expect(calls).toBe(3);
      expect(recovery.state).toBe('running'); expect(recovery.current).toBe(pool.created[1]);
      releaseHung(); await flush();
      expect(pool.created[2].disposed).toBe(1); // the hung attempt's owner arrived late
      expect(recovery.current).toBe(pool.created[1]);
      expect(recovery.diagnostics().events.find(event => event.kind === 'attempt-failed')?.error)
        .toBe('DEVICE_RECOVERY_TIMEOUT: attempt 1 exceeded 500 ms');
    } finally { vi.useRealTimers(); }
  });

  it('stops after too many losses inside the window and accepts them again once it slides', async () => {
    vi.useFakeTimers();
    try {
      const pool = owners(), failed = vi.fn();
      const recovery = new GpuDeviceRecovery(async () => pool.make(), { failed },
        { maxLossesPerWindow: 2, windowMs: 10_000 }, () => Date.now());
      await recovery.start();
      pool.created[0].lose(); await flush(); expect(recovery.diagnostics().recoveries).toBe(1);
      await flush(10_001);
      pool.created[1].lose(); await flush(); expect(recovery.diagnostics().recoveries).toBe(2);
      pool.created[2].lose(); await flush(); expect(recovery.diagnostics().recoveries).toBe(3);
      pool.created[3].lose(); await flush();
      expect(recovery.state).toBe('failed');
      expect(recovery.diagnostics().failure).toMatch(/^DEVICE_RECOVERY_LIMIT: 3 device losses within 10000 ms exceed 2; last: destroyed/);
      expect(failed).toHaveBeenCalledOnce();
      expect(pool.created).toHaveLength(4); expect(pool.created[3].disposed).toBe(1);
    } finally { vi.useRealTimers(); }
  });

  it('cancels an in-flight rebuild on dispose and releases the owner it produces', async () => {
    vi.useFakeTimers();
    try {
      const pool = owners(), recovered = vi.fn();
      let finish!: () => void, calls = 0;
      const recovery = new GpuDeviceRecovery(() => calls++ === 0 ? Promise.resolve(pool.make())
        : new Promise<FakeOwner>(resolve => { finish = () => resolve(pool.make()); }), { recovered }, {}, () => Date.now());
      (await recovery.start()).lose(); await flush();
      recovery.dispose(); finish(); await flush();
      expect(recovered).not.toHaveBeenCalled(); expect(pool.created[1].disposed).toBe(1);
      expect(recovery.state).toBe('disposed'); expect(recovery.current).toBeUndefined();
    } finally { vi.useRealTimers(); }
  });

  it('does not retry initial creation and validates every policy bound', async () => {
    const recovery = new GpuDeviceRecovery<FakeOwner>(async () => { throw new Error('WEBGPU_UNAVAILABLE'); });
    await expect(recovery.start()).rejects.toThrow('WEBGPU_UNAVAILABLE');
    expect(recovery.state).toBe('failed'); expect(recovery.diagnostics().failure).toBe('WEBGPU_UNAVAILABLE');
    await expect(recovery.start()).rejects.toThrow('DEVICE_RECOVERY_STATE');
    expect(resolveDeviceRecoveryPolicy()).toEqual({ maxAttempts: 3, backoffMs: [0, 250, 1000], attemptTimeoutMs: 20_000,
      maxLossesPerWindow: 3, windowMs: 60_000 });
    for (const policy of [{ maxAttempts: 0 }, { maxAttempts: 9 }, { backoffMs: [] }, { backoffMs: [-1] }, { backoffMs: [0.5] },
      { backoffMs: new Array<number>(9).fill(0) }, { attemptTimeoutMs: 99 }, { maxLossesPerWindow: 17 }, { windowMs: 999 }]) {
      expect(() => resolveDeviceRecoveryPolicy(policy)).toThrow('DEVICE_RECOVERY_POLICY');
    }
  });
});
