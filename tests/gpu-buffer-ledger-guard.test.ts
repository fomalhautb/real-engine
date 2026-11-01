import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/**
 * `name.destroy()` followed, within the same or the next line, by a read of `name.size`, and
 * `name.dispose()` followed by `name.ownedBytes` (a fixture getter that sums buffer sizes).
 */
function sizeReadsAfterDestroy(text: string): string[] {
  const found: string[] = [];
  for (const match of text.matchAll(/([A-Za-z_$][\w$]*(?:\??\.[A-Za-z_$][\w$]*)*)\??\.(destroy|dispose)\(\)/g)) {
    const name = match[1], start = match.index + match[0].length;
    const lineEnd = text.indexOf('\n', text.indexOf('\n', start) + 1);
    const following = text.slice(start, lineEnd < 0 ? undefined : lineEnd);
    const escaped = name.replace(/[.?$]/g, character => '\\' + character);
    const property = match[2] === 'destroy' ? 'size' : 'ownedBytes';
    if (new RegExp(`(?<![\\w$.])${escaped}\\??\\.${property}\\b`).test(following)) found.push(name);
  }
  return found;
}

describe('GPU buffer ledgers', () => {
  it('read GPUBuffer.size before destroy() in the engine and the browser fixtures', () => {
    // WebKit 27.2 and Safari report GPUBuffer.size as 0 once the buffer is destroyed (Chrome keeps
    // the creation size, as the WebGPU specification requires). A ledger that subtracts the size
    // after destroy() never returns to zero there; read the size first.
    const offenders: string[] = [];
    for (const directory of ['../src', './browser']) {
      const root = fileURLToPath(new URL(directory, import.meta.url));
      const visit = (path: string): void => {
        for (const entry of readdirSync(path, { withFileTypes: true })) {
          const child = join(path, entry.name);
          if (entry.isDirectory()) { visit(child); continue; }
          if (!entry.name.endsWith('.ts')) continue;
          for (const name of sizeReadsAfterDestroy(readFileSync(child, 'utf8'))) offenders.push(`${relative(root, child)}: ${name}`);
        }
      };
      visit(root);
    }
    expect(offenders).toEqual([]);
  });

  it('recognises the pattern it guards against', () => {
    expect(sizeReadsAfterDestroy('buffer.destroy(); bytes -= buffer.size;')).toEqual(['buffer']);
    expect(sizeReadsAfterDestroy('this.buffers.splice(i, 1); buffer.destroy();\n  if (live) this.owned -= buffer.size;')).toEqual(['buffer']);
    expect(sizeReadsAfterDestroy('ready.output.destroy(); ready.instances.destroy();\n  account(-ready.output.size - ready.instances.size);'))
      .toEqual(['ready.output', 'ready.instances']);
    expect(sizeReadsAfterDestroy('bytes -= buffer.size; buffer.destroy();')).toEqual([]);
    expect(sizeReadsAfterDestroy('other.destroy(); bytes -= buffer.size;')).toEqual([]);
    expect(sizeReadsAfterDestroy('buffer.destroy();\n\n  later(buffer.size);')).toEqual([]);
    expect(sizeReadsAfterDestroy('arm.udf.dispose(); account(-arm.udf.ownedBytes);')).toEqual(['arm.udf']);
    expect(sizeReadsAfterDestroy('account(-arm.udf.ownedBytes); arm.udf.dispose();')).toEqual([]);
    expect(sizeReadsAfterDestroy('renderer.dispose(); expect(renderer.stats.ownedGpuBytes).toBe(0);')).toEqual([]);
  });
});
