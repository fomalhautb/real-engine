import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { TEXTURE_COMPRESSION_FEATURES } from '../src/texture-storage';

// Safari 26.6.2 on the Apple M3 Max reports no subgroups, clip-distances, primitive-index or
// texture-formats-tier2 (2026-10-31 safaridriver probe); Playwright WebKit 27.2 and Chrome 156 do.
// The engine needs none of them today. These guards keep it that way until a module adds an
// explicit capability check with a typed error or a documented fallback for the missing feature.
const SAFARI_MISSING = ['subgroups', 'clip-distances', 'primitive-index', 'texture-formats-tier2'] as const;
// Core WebGPU storage-texture formats; read_write access is core only for the r32 formats.
const CORE_STORAGE_FORMATS = new Set(['rgba8unorm', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'rgba16uint', 'rgba16sint', 'rgba16float',
  'r32uint', 'r32sint', 'r32float', 'rg32uint', 'rg32sint', 'rg32float', 'rgba32uint', 'rgba32sint', 'rgba32float']);
const CORE_READ_WRITE_FORMATS = new Set(['r32uint', 'r32sint', 'r32float']);

const root = fileURLToPath(new URL('../src', import.meta.url));
function sources(): { name: string, text: string }[] {
  const found: { name: string, text: string }[] = [];
  const visit = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) { visit(path); continue; }
      if (entry.name.endsWith('.ts') || entry.name.endsWith('.wgsl')) found.push({ name: relative(root, path), text: readFileSync(path, 'utf8') });
    }
  };
  visit(root);
  return found;
}

describe('optional WebGPU features', () => {
  it('requests only timestamp-query and texture compression, each after an adapter check', () => {
    const renderer = readFileSync(join(root, 'renderer.ts'), 'utf8');
    const requests = renderer.match(/requiredFeatures\b[^;]*;/g) ?? [];
    expect(requests).toEqual([
      "requiredFeatures: GPUFeatureName[] = timing ? ['timestamp-query'] : [];",
      'requiredFeatures.push(feature);',
      'requiredFeatures });',
    ]);
    expect(renderer).toContain("const timing = adapter.features.has('timestamp-query');");
    expect(renderer).toContain('for (const feature of TEXTURE_COMPRESSION_FEATURES) if (adapter.features.has(feature)) requiredFeatures.push(feature);');
    expect(TEXTURE_COMPRESSION_FEATURES).toEqual(['texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc']);
    const offenders = sources().filter(source => source.name !== 'renderer.ts' && /\brequiredFeatures\b|\brequestDevice\s*\(/.test(source.text));
    expect(offenders.map(source => source.name)).toEqual([]);
  });

  it('keeps features Safari lacks out of every engine module and shader', () => {
    const offenders: string[] = [];
    for (const { name, text } of sources()) {
      for (const feature of SAFARI_MISSING) if (text.includes(`'${feature}'`) || text.includes(`"${feature}"`)) offenders.push(`${name}: ${feature}`);
      for (const match of text.matchAll(/\benable\s+([a-z_0-9, ]+);/g)) offenders.push(`${name}: enable ${match[1]}`);
      for (const match of text.matchAll(/\b(subgroup[A-Z]\w*|quad[A-Z]\w*)\s*\(|@builtin\(\s*(subgroup_size|subgroup_invocation_id|clip_distances|primitive_index)\s*\)/g)) {
        offenders.push(`${name}: ${match[1] ?? match[2]}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('uses only core storage-texture formats and access modes', () => {
    const uses: string[] = [], offenders: string[] = [];
    for (const { name, text } of sources()) {
      for (const match of text.matchAll(/texture_storage_\w+\s*<\s*(\w+)\s*,\s*(\w+)\s*>/g)) {
        uses.push(`${name}: ${match[1]} ${match[2]}`);
        if (!CORE_STORAGE_FORMATS.has(match[1]) || (match[2] === 'read_write' && !CORE_READ_WRITE_FORMATS.has(match[1]))) offenders.push(`${name}: ${match[1]} ${match[2]}`);
      }
      for (const match of text.matchAll(/storageTexture:\s*\{\s*access:\s*'([\w-]+)',\s*format:\s*'(\w+)'/g)) {
        uses.push(`${name}: ${match[2]} ${match[1]}`);
        if (!CORE_STORAGE_FORMATS.has(match[2]) || (match[1] === 'read-write' && !CORE_READ_WRITE_FORMATS.has(match[2]))) offenders.push(`${name}: ${match[2]} ${match[1]}`);
      }
    }
    expect(offenders).toEqual([]);
    // The occlusion pyramid is the only storage texture today; a new one is reviewed here.
    expect(uses.sort()).toEqual(['gpu-occlusion.ts: r32float write-only', 'shaders/occlusion-pyramid.wgsl: r32float write', 'shaders/occlusion-reduce.wgsl: r32float write']);
  });
});
