/**
 * Portable multi-pass GPU timestamp spans.
 *
 * A span times several consecutive passes with one slot pair: the first pass records the span's
 * beginning and the last pass its end. WebGPU lets a pass write only one of
 * `beginningOfPassWriteIndex` / `endOfPassWriteIndex`, but on the target Apple M3 Max (macOS 26.6.2,
 * Chrome 156, Dawn Metal backend) a frame whose compute passes carry such partial writes fails its
 * Metal command buffer with `kIOGPUCommandBufferCallbackErrorOutOfMemory` ("Insufficient Memory")
 * and the device is lost. Stripping only the partial writes, or completing them, removes the loss.
 *
 * Every timed pass therefore writes both of its ends. The end a pass does not own goes to one scratch
 * slot that the query set reserves after its measured slots; the scratch slot is never resolved or
 * read, so a span still measures exactly the first pass's beginning to the last pass's end.
 */

const scratchSlots = new WeakMap<GPUQuerySet, number>();
const MAX_QUERY_SET_COUNT = 4096;

/** Creates a timestamp query set with `measuredSlots` addressable slots plus one reserved scratch slot. */
export function createTimestampQuerySet(device: GPUDevice, measuredSlots: number, label = 'pass timestamps'): GPUQuerySet {
  if (!Number.isSafeInteger(measuredSlots) || measuredSlots < 2 || measuredSlots + 1 > MAX_QUERY_SET_COUNT) {
    throw new RangeError('TIMESTAMP_SLOTS: measured slot count must be an integer in [2, ' + (MAX_QUERY_SET_COUNT - 1) + ']');
  }
  const querySet = device.createQuerySet({ label, type: 'timestamp', count: measuredSlots + 1 });
  scratchSlots.set(querySet, measuredSlots);
  return querySet;
}

function scratchFor(span: GPUComputePassTimestampWrites): number {
  const scratch = scratchSlots.get(span.querySet);
  if (scratch === undefined) throw new Error('TIMESTAMP_SPAN_SCRATCH: span query sets must come from createTimestampQuerySet');
  const { beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } = span;
  if (begin === undefined || end === undefined || begin === end || begin >= scratch || end >= scratch) {
    throw new RangeError('TIMESTAMP_SPAN_SLOTS: a span needs distinct measured beginning and end slots');
  }
  return scratch;
}

/** Writes for the first pass of a span: its beginning is the span's beginning, its end is scratch. */
export function timestampSpanStart(span?: GPUComputePassTimestampWrites): GPUComputePassTimestampWrites | undefined {
  return span && { querySet: span.querySet, beginningOfPassWriteIndex: span.beginningOfPassWriteIndex, endOfPassWriteIndex: scratchFor(span) };
}

/** Writes for the last pass of a span: its beginning is scratch, its end is the span's end. */
export function timestampSpanEnd(span?: GPUComputePassTimestampWrites): GPUComputePassTimestampWrites | undefined {
  return span && { querySet: span.querySet, beginningOfPassWriteIndex: scratchFor(span), endOfPassWriteIndex: span.endOfPassWriteIndex };
}
