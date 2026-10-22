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
