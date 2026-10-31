import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { crc32, deflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { buildMipChain } from '../src/texture-assets';
import { decodeBc7Image } from '../src/texture-bc7';

const script = resolve('scripts/build-texture.mjs');
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

function run(input: string, output: string, extra: string[]): string {
  return execFileSync(process.execPath, [script, '--input', input, '--output', output, ...extra], {
    cwd: resolve('.'), windowsHide: true, timeout: 120_000, maxBuffer: 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'],
  }).toString();
}

function chunk(type: string, data: Uint8Array): Buffer {
  const header = Buffer.alloc(8), crc = Buffer.alloc(4);
  header.writeUInt32BE(data.length, 0);
  header.write(type, 4, 'ascii');
  crc.writeUInt32BE(crc32(Buffer.concat([Buffer.from(type, 'ascii'), data])), 0);
  return Buffer.concat([header, data, crc]);
}

/** RGBA8 PNG with filter 0 rows, written independently of the decoder under test. */
function png(width: number, height: number, pixels: Uint8Array): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 6;
  const raw = Buffer.alloc(height * (width * 4 + 1));
  for (let y = 0; y < height; y++) raw.set(pixels.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))]);
}

function pixels(width: number, height: number, alpha: boolean): Uint8Array {
  const result = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const o = (y * width + x) * 4;
    result[o] = x * 19 & 255; result[o + 1] = y * 23 & 255; result[o + 2] = (x * y * 7 + 40) & 255;
    result[o + 3] = alpha ? (x * 11 + y * 5) & 255 : 255;
  }
  return result;
}

async function files(root: string, prefix = ''): Promise<Map<string, Buffer>> {
  const result = new Map<string, Buffer>();
  for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) for (const [key, bytes] of await files(root, name)) result.set(key, bytes);
    else result.set(name, await readFile(join(root, name)));
  }
  return result;
}

async function withRoot(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(resolve('.'), '.texture-test-'));
  try { await body(root); } finally {
    if (dirname(root) !== resolve('.') || !basename(root).startsWith('.texture-test-')) throw Error('refusing to remove');
    await rm(root, { recursive: true, force: true });
  }
}

describe('texture CLI builtin encoder', () => {
  it('builds identical BC7 packages for any worker count with a truthful quality record', async () => {
    await withRoot(async root => {
      const width = 20, height = 12, source = pixels(width, height, true);
      await writeFile(join(root, 'albedo.png'), png(width, height, source));
      const common = ['--id', 'albedo', '--format', 'bc7', '--color-space', 'srgb', '--encoder', 'builtin', '--fallback-size', '8'];
      const single = JSON.parse(run(join(root, 'albedo.png'), join(root, 'single'), [...common, '--workers', '0']));
      run(join(root, 'albedo.png'), join(root, 'pooled'), [...common, '--workers', '3']);
      const a = await files(join(root, 'single')), b = await files(join(root, 'pooled'));
      expect([...a.keys()].sort()).toEqual(['build.json', 'manifest.json', 'mips/0.bin', 'mips/1.bin', 'mips/2.bin', 'mips/3.bin', 'mips/4.bin', 'source.dds', 'source.png']);
      for (const [name, bytes] of a) expect(bytes.equals(b.get(name)!), name).toBe(true);
      expect(single.run.workers).toBe(0);

      const manifest = JSON.parse(a.get('manifest.json')!.toString());
      expect(manifest).toMatchObject({ version: 1, id: 'albedo', width, height, format: 'bc7', colorSpace: 'srgb' });
      // 20x12 -> 10x6 -> 5x3 -> 2x1 -> 1x1; only level 0 has whole 4x4 blocks, so it is the fallback.
      expect(manifest.mips.map((mip: { width: number; height: number; byteLength: number }) => [mip.width, mip.height, mip.byteLength]))
        .toEqual([[20, 12, 240], [10, 6, 96], [5, 3, 32], [2, 1, 16], [1, 1, 16]]);
      expect(manifest.fallbackMip).toBe(0);
      const reference = buildMipChain({ width, height, pixels: source, colorSpace: 'srgb' });
      const build = JSON.parse(a.get('build.json')!.toString());
      expect(build.source).toEqual({ file: 'source.png', byteLength: a.get('source.png')!.length, sha256: sha256(a.get('source.png')!) });
      expect(build.codec.name).toBe('real-engine builtin');
      expect(build.codec.encoder).toMatchObject({ format: 'bc7', quality: 2, earlyOutError: 96 });
      expect(build.codec.decodedSource.rgba8Sha256).toBe(sha256(source));
      expect(build.codec.timingsMs).toBeUndefined();
      expect(Object.keys(build.codec.moduleSha256)).toContain('src/texture-bc7.ts');
      manifest.mips.forEach((mip: { width: number; height: number; sha256: string }, level: number) => {
        const blocks = a.get(`mips/${level}.bin`)!;
        expect(sha256(blocks)).toBe(mip.sha256);
        const decoded = decodeBc7Image(blocks, mip.width, mip.height), expected = reference[level].pixels;
        const maxAbs = [0, 0, 0, 0];
        for (let i = 0; i < decoded.length; i++) maxAbs[i & 3] = Math.max(maxAbs[i & 3], Math.abs(decoded[i] - expected[i]));
        expect(build.codec.quality.levels[level].maxAbs).toEqual(maxAbs);
        expect(build.codec.quality.levels[level].modes.reduce((n: number, v: number) => n + v, 0)).toBe(blocks.length / 16);
      });
    });
  });

  it('stores RGBA8 reference levels exactly equal to buildMipChain, also from TGA', async () => {
    await withRoot(async root => {
      const width = 6, height = 4, source = pixels(width, height, false);
      await writeFile(join(root, 'normal.png'), png(width, height, source));
      const tga = new Uint8Array(18 + width * height * 4);
      tga[2] = 2; tga[12] = width; tga[14] = height; tga[16] = 32; tga[17] = 0x28;
      for (let i = 0; i < width * height; i++) tga.set([source[i * 4 + 2], source[i * 4 + 1], source[i * 4], source[i * 4 + 3]], 18 + i * 4);
      await writeFile(join(root, 'normal.tga'), tga);
      for (const input of ['normal.png', 'normal.tga']) {
        const output = join(root, 'out-' + input);
        run(join(root, input), output, ['--format', 'rgba8', '--color-space', 'linear', '--encoder', 'builtin']);
        const reference = buildMipChain({ width, height, pixels: source, colorSpace: 'linear' });
        const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
        expect(manifest.mips.length).toBe(reference.length);
        for (let level = 0; level < reference.length; level++) {
          expect((await readFile(join(output, `mips/${level}.bin`))).equals(Buffer.from(reference[level].pixels))).toBe(true);
        }
        const build = JSON.parse(await readFile(join(output, 'build.json'), 'utf8'));
        expect(build.codec.encoder).toEqual({ format: 'rgba8', lossless: true });
        expect(build.codec.quality).toBeNull();
      }
    });
  });

  it('rejects inconsistent encoder options and non-multiple-of-4 BC7 sources', async () => {
    await withRoot(async root => {
      await writeFile(join(root, 'odd.png'), png(6, 4, pixels(6, 4, false)));
      const input = join(root, 'odd.png');
      expect(() => run(input, join(root, 'a'), ['--format', 'bc7', '--color-space', 'srgb', '--encoder', 'builtin'])).toThrow();
      expect(() => run(input, join(root, 'b'), ['--format', 'rgba8', '--color-space', 'srgb', '--workers', '2'])).toThrow();
      expect(() => run(input, join(root, 'c'), ['--format', 'rgba8', '--color-space', 'srgb', '--encoder', 'builtin', '--texconv', input])).toThrow();
      expect(() => run(input, join(root, 'd'), ['--format', 'rgba8', '--color-space', 'srgb', '--encoder', 'builtin', '--bc7-quality', '3'])).toThrow();
      expect(() => run(input, join(root, 'e'), ['--format', 'rgba8', '--color-space', 'srgb', '--encoder', 'magic'])).toThrow();
      expect((await readdir(root)).sort()).toEqual(['odd.png']);
    });
  });
});
