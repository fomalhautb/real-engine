import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { createTimestampQuerySet, timestampSpanEnd, timestampSpanStart } from '../src/gpu-timestamp-span';

function fakeDevice() {
  const created: GPUQuerySetDescriptor[] = [];
  const device = { createQuerySet(descriptor: GPUQuerySetDescriptor) { created.push(descriptor); return { count: descriptor.count } as GPUQuerySet; } };
  return { device: device as unknown as GPUDevice, created };
}

describe('portable multi-pass timestamp spans', () => {
  it('reserves one scratch slot after the measured slots', () => {
    const { device, created } = fakeDevice();
    createTimestampQuerySet(device, 12);
    expect(created).toEqual([{ label: 'pass timestamps', type: 'timestamp', count: 13 }]);
    for (const count of [0, 1, 2.5, Number.NaN, 4096]) expect(() => createTimestampQuerySet(device, count)).toThrow('TIMESTAMP_SLOTS');
    expect(created).toHaveLength(1);
  });

  it('gives the first and last pass of a span both writes, sending the unowned end to scratch', () => {
    const querySet = createTimestampQuerySet(fakeDevice().device, 6);
    const span = { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
    expect(timestampSpanStart(span)).toEqual({ querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 6 });
    expect(timestampSpanEnd(span)).toEqual({ querySet, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 3 });
    expect(timestampSpanStart(undefined)).toBeUndefined();
    expect(timestampSpanEnd(undefined)).toBeUndefined();
  });

  it('rejects query sets without a reserved scratch slot and spans that are not two measured slots', () => {
    const unreserved = { count: 8 } as GPUQuerySet;
    expect(() => timestampSpanStart({ querySet: unreserved, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 })).toThrow('TIMESTAMP_SPAN_SCRATCH');
    const querySet = createTimestampQuerySet(fakeDevice().device, 4);
    for (const span of [{ beginningOfPassWriteIndex: 0 }, { endOfPassWriteIndex: 1 }, { beginningOfPassWriteIndex: 1, endOfPassWriteIndex: 1 },
      { beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 4 }, { beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 3 }]) {
      expect(() => timestampSpanStart({ querySet, ...span })).toThrow('TIMESTAMP_SPAN_SLOTS');
      expect(() => timestampSpanEnd({ querySet, ...span })).toThrow('TIMESTAMP_SPAN_SLOTS');
    }
  });

  it('keeps pass write indices out of every module except the span helper and the renderer slot allocator', () => {
    // A pass that writes only one end of its timestampWrites loses the device on the Apple/Metal
    // target. Modules forward whole slot pairs or use timestampSpanStart / timestampSpanEnd.
    const root = fileURLToPath(new URL('../src', import.meta.url)), offenders: string[] = [];
    const visit = (directory: string): void => {
      for (const entry of readdirSync(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) { visit(path); continue; }
        if (!entry.name.endsWith('.ts')) continue;
        const name = relative(root, path), uses = readFileSync(path, 'utf8').match(/\b(?:beginningOfPassWriteIndex|endOfPassWriteIndex)\b/g) ?? [];
        if (name === 'gpu-timestamp-span.ts') continue;
        if (name === 'renderer.ts' ? uses.length !== 2 : uses.length) offenders.push(name + ' (' + uses.length + ')');
      }
    };
    visit(root);
    expect(offenders).toEqual([]);
  });
});
