import { describe, expect, it, vi } from 'vitest';
import { subscribeGpuDeviceLoss } from '../src/gpu-device-loss';

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
