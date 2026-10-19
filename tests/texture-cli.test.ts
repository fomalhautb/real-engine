import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const script = resolve('scripts/build-texture.mjs');
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

type Format = 'rgba8' | 'bc7';
type Mip = { level: number; width: number; height: number; bytes: Buffer };
type Manifest = {
  version: number;
  id: string;
  width: number;
  height: number;
  colorSpace: string;
  format: string;
  fallbackMip: number;
  mips: { level: number; width: number; height: number; byteLength: number; sha256: string; file: string }[];
};

// This fixture writes the DDS and DX10 headers directly. None of its expected
// mip offsets, lengths, or hashes come from the texture parser under test.
function dds(width: number, height: number, format: Format, colorSpace: 'srgb' | 'linear',
  options: { truncatedTail?: boolean; cube?: boolean; legacy?: boolean } = {}): { bytes: Buffer; mips: Mip[] } {
  const mips: Mip[] = [];
  let w = width;
  let h = height;
  do {
    const byteLength = format === 'rgba8'
      ? w * h * 4
      : Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    const bytes = Buffer.alloc(byteLength);
    for (let i = 0; i < byteLength; i++) bytes[i] = (mips.length * 71 + i * 17 + 3) & 255;
    mips.push({ level: mips.length, width: w, height: h, bytes });
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  } while (w !== mips.at(-1)!.width || h !== mips.at(-1)!.height);

  const header = Buffer.alloc(148);
  header.write('DDS ', 0, 'ascii');
  header.writeUInt32LE(124, 4);
  header.writeUInt32LE(format === 'rgba8' ? 0x0002100f : 0x000a1007, 8);
  // caps, dimensions, pixel format, mip count, and pitch or linear size
  header.writeUInt32LE(height, 12);
  header.writeUInt32LE(width, 16);
  header.writeUInt32LE(format === 'rgba8' ? width * 4
    : Math.ceil(width / 4) * Math.ceil(height / 4) * 16, 20);
  header.writeUInt32LE(mips.length, 28);
  header.writeUInt32LE(32, 76); // DDS_PIXELFORMAT size
  header.writeUInt32LE(0x4, 80); // DDPF_FOURCC
  header.write(options.legacy ? 'DXT1' : 'DX10', 84, 'ascii');
  header.writeUInt32LE(0x401008, 108); // texture, complex, mipmap
  if (options.cube) header.writeUInt32LE(0x200, 112);
  header.writeUInt32LE(format === 'rgba8' ? (colorSpace === 'srgb' ? 29 : 28)
    : (colorSpace === 'srgb' ? 99 : 98), 128);
  header.writeUInt32LE(3, 132); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
  header.writeUInt32LE(options.cube ? 0x4 : 0, 136);
  header.writeUInt32LE(1, 140); // array size
  const payload = Buffer.concat(mips.map(mip => mip.bytes));
  return { bytes: Buffer.concat([header, options.truncatedTail ? payload.subarray(0, -1) : payload]), mips };
}

function run(input: string, output: string, extra: string[] = []): void {
  execFileSync(process.execPath, [script, '--input', input, '--output', output, ...extra], {
    cwd: resolve('.'), windowsHide: true, timeout: 120_000, maxBuffer: 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

async function removeOwnTestRoot(root: string): Promise<void> {
  const absolute = resolve(root);
  if (dirname(absolute) !== resolve('.') || !basename(absolute).startsWith('.texture-test-')) {
    throw new Error('refusing to remove a directory outside this test root');
  }
  await rm(absolute, { recursive: true, force: true });
}

async function files(root: string, prefix = ''): Promise<Map<string, Buffer>> {
  const result = new Map<string, Buffer>();
  for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) {
      for (const [key, bytes] of await files(root, name)) result.set(key, bytes);
    } else result.set(name, await readFile(join(root, name)));
  }
  return result;
}

async function verifyOutput(output: string, source: Buffer, expected: Mip[],
  metadata: Pick<Manifest, 'id' | 'width' | 'height' | 'format' | 'colorSpace' | 'fallbackMip'>): Promise<void> {
  const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8')) as Manifest;
  expect(manifest).toEqual({
    version: 1,
    ...metadata,
    mips: expected.map(mip => ({
      level: mip.level, width: mip.width, height: mip.height,
      byteLength: mip.bytes.length, sha256: sha256(mip.bytes), file: `mips/${mip.level}.bin`,
    })),
  });
  for (const mip of expected) expect(await readFile(join(output, `mips/${mip.level}.bin`))).toEqual(mip.bytes);
  expect(await readFile(join(output, 'source.dds'))).toEqual(source);
  const build = JSON.parse(await readFile(join(output, 'build.json'), 'utf8')) as Record<string, unknown>;
  expect(build.source).toEqual({ file: 'source.dds', byteLength: source.length, sha256: sha256(source) });
  expect(build.encoded).toEqual({ file: 'source.dds', byteLength: source.length, sha256: sha256(source) });
  expect(build.nativeBytes).toBe(expected.reduce((sum, mip) => sum + mip.bytes.length, 0));
  expect(build.fallbackBytes).toBe(expected.slice(metadata.fallbackMip)
    .reduce((sum, mip) => sum + mip.bytes.length, 0));
  expect(build.codec).toBeNull();
  expect([...await files(output)].map(([name]) => name).sort()).toEqual([
    'build.json', 'manifest.json', ...expected.map(mip => `mips/${mip.level}.bin`), 'source.dds',
  ].sort());
}

describe('texture asset CLI', () => {
  it('packages each RGBA8 mip, source provenance, and a size-limited fallback deterministically', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.texture-test-'));
    try {
      const fixture = dds(8, 4, 'rgba8', 'srgb');
      const input = join(temporary, 'source with spaces.dds');
      await writeFile(input, fixture.bytes);
      const first = join(temporary, 'first output');
      const second = join(temporary, 'second output');
      run(input, first, ['--id', 'texture.sample', '--fallback-size', '2']);
      run(input, second, ['--id', 'texture.sample', '--fallback-size', '2']);
      await verifyOutput(first, fixture.bytes, fixture.mips, {
        id: 'texture.sample', width: 8, height: 4, format: 'rgba8', colorSpace: 'srgb', fallbackMip: 2,
      });
      const left = await files(first);
      const right = await files(second);
      expect([...left.keys()].sort()).toEqual([...right.keys()].sort());
      for (const [name, bytes] of left) expect(bytes).toEqual(right.get(name));
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('uses the file stem and default fallback size, and keeps the BC7 tail intact', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.texture-test-'));
    try {
      const fixture = dds(128, 64, 'bc7', 'linear');
      const input = join(temporary, 'stone wall.dds');
      const output = join(temporary, 'stream');
      await writeFile(input, fixture.bytes);
      run(input, output);
      await verifyOutput(output, fixture.bytes, fixture.mips, {
        id: 'stone wall', width: 128, height: 64, format: 'bc7', colorSpace: 'linear', fallbackMip: 1,
      });
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('chooses the coarsest block-aligned BC7 mip when none fits the requested size', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.texture-test-'));
    try {
      const fixture = dds(16, 16, 'bc7', 'srgb');
      const input = join(temporary, 'blocks.dds');
      const output = join(temporary, 'stream');
      await writeFile(input, fixture.bytes);
      run(input, output, ['--fallback-size', '2']);
      await verifyOutput(output, fixture.bytes, fixture.mips, {
        id: 'blocks', width: 16, height: 16, format: 'bc7', colorSpace: 'srgb', fallbackMip: 2,
      });
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('does not overwrite an output directory or publish malformed DDS inputs', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.texture-test-'));
    try {
      const good = dds(8, 8, 'rgba8', 'linear');
      const input = join(temporary, 'good.dds');
      await writeFile(input, good.bytes);
      const existing = join(temporary, 'existing');
      await mkdir(existing);
      await writeFile(join(existing, 'sentinel.txt'), 'keep me');
      expect(() => run(input, existing)).toThrow();
      expect(await readFile(join(existing, 'sentinel.txt'), 'utf8')).toBe('keep me');

      const invalid: [string, Buffer, string[]][] = [
        ['truncated', dds(8, 8, 'rgba8', 'linear', { truncatedTail: true }).bytes, []],
        ['legacy', dds(8, 8, 'rgba8', 'linear', { legacy: true }).bytes, []],
        ['cube', dds(8, 8, 'rgba8', 'linear', { cube: true }).bytes, []],
        ['too-wide', dds(8193, 1, 'rgba8', 'linear').bytes, []],
        ['bad-id', good.bytes, ['--id', 'bad\nid']],
        ['bad-fallback', good.bytes, ['--fallback-size', '0']],
        ['format-mismatch', good.bytes, ['--format', 'bc7']],
        ['color-mismatch', good.bytes, ['--color-space', 'srgb']],
      ];
      for (const [name, bytes, extra] of invalid) {
        const badInput = join(temporary, `${name}.dds`);
        const output = join(temporary, `${name}-output`);
        await writeFile(badInput, bytes);
        expect(() => run(badInput, output, extra)).toThrow();
        expect((await readdir(temporary)).includes(`${name}-output`)).toBe(false);
      }
      expect((await readdir(temporary)).every(name => !/^\..*\.tmp-/.test(name))).toBe(true);
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('requires explicit PNG conversion options and rejects a malformed PNG before publishing', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.texture-test-'));
    try {
      const input = join(temporary, 'image.png');
      await writeFile(input, Buffer.from('not a PNG'));
      const output = join(temporary, 'png-output');
      expect(() => run(input, output)).toThrow();
      expect(() => run(input, output, [
        '--format', 'bc7', '--color-space', 'srgb', '--texconv', join(temporary, 'missing-texconv.exe'),
      ])).toThrow();
      expect((await readdir(temporary)).includes('png-output')).toBe(false);

      const plausible = Buffer.alloc(33);
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(plausible);
      plausible.writeUInt32BE(13, 8);
      plausible.write('IHDR', 12, 'ascii');
      plausible.writeUInt32BE(4, 16);
      plausible.writeUInt32BE(4, 20);
      plausible[24] = 8;
      plausible[25] = 6;
      await writeFile(input, plausible);
      expect(() => run(input, output, [
        '--format', 'rgba8', '--color-space', 'linear', '--texconv', join(temporary, 'missing-texconv.exe'),
      ])).toThrow();
      expect((await readdir(temporary)).includes('png-output')).toBe(false);
      expect((await readdir(temporary)).every(name => !/^\..*\.tmp-/.test(name))).toBe(true);
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);
});
