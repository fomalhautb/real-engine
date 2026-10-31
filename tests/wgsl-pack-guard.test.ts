import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

describe('WGSL normalized packing', () => {
  it('keeps pack*norm builtins out of every engine shader', () => {
    // On the Apple/Metal target, a pack2x16snorm / pack2x16unorm / pack4x8snorm / pack4x8unorm
    // result outside lane x of an integer fragment output zeroes or scrambles the whole output
    // (raster receivers decoded every normal as +Z). WGSL helpers are shared across stages, so the
    // builtins are kept out entirely; surfacePackSnorm2x16 packs with integer ops instead.
    const root = fileURLToPath(new URL('../src', import.meta.url)), offenders: string[] = [];
    const visit = (directory: string): void => {
      for (const entry of readdirSync(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) { visit(path); continue; }
        if (!entry.name.endsWith('.ts') && !entry.name.endsWith('.wgsl')) continue;
        const uses = readFileSync(path, 'utf8').match(/(?<![\w.])pack(?:2x16|4x8)(?:snorm|unorm)\s*\(/g) ?? [];
        if (uses.length) offenders.push(relative(root, path) + ' (' + uses.length + ')');
      }
    };
    visit(root);
    expect(offenders).toEqual([]);
    expect(readFileSync(join(root, 'shaders', 'surface-frame.wgsl'), 'utf8')).toContain('fn surfacePackSnorm2x16(value: vec2f) -> u32');
  });
});
