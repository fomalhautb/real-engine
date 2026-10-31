import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { crc32, deflateRawSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

const script = resolve('scripts/extract-archive.mjs');
const { safeEntryName } = await import(pathToFileURL(script).href) as { safeEntryName: (name: string) => boolean };
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

/** Minimal independent ZIP writer (stored or raw-deflate entries, no extras). */
function zip(entries: { name: string; data: Buffer; deflate?: boolean; corrupt?: boolean }[]): Buffer {
  const locals: Buffer[] = [], centrals: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, 'utf8'), payload = entry.deflate ? deflateRawSync(entry.data) : Buffer.from(entry.data);
    if (entry.corrupt) payload[0] ^= 0xff;
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(entry.deflate ? 8 : 0, 8);
    local.writeUInt32LE(crc32(entry.data), 14); local.writeUInt32LE(payload.length, 18); local.writeUInt32LE(entry.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6);
    central.writeUInt16LE(entry.deflate ? 8 : 0, 10); central.writeUInt32LE(crc32(entry.data), 16);
    central.writeUInt32LE(payload.length, 20); central.writeUInt32LE(entry.data.length, 24); central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, name, payload); centrals.push(central, name);
    offset += 30 + name.length + payload.length;
  }
  const directory = Buffer.concat(centrals), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}

const run = (args: string[]): string => execFileSync(process.execPath, [script, ...args], {
  cwd: resolve('.'), windowsHide: true, timeout: 60_000, maxBuffer: 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'],
}).toString();

async function withRoot(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(resolve('.'), '.extract-archive-test-'));
  try { await body(root); } finally {
    if (dirname(root) !== resolve('.') || !basename(root).startsWith('.extract-archive-test-')) throw Error('refusing to remove');
    await rm(root, { recursive: true, force: true });
  }
}

describe('extract-archive', () => {
  it('accepts only relative, normalized entry names', () => {
    for (const name of ['scene/a.gltf', 'a', 'dir/']) expect(safeEntryName(name), name).toBe(true);
    for (const name of ['../a', 'a/../b', '/abs', 'C:/x', 'a\\b', 'a//b', './a', 'a\u0001']) expect(safeEntryName(name), name).toBe(false);
  });

  it('extracts a glTF with its dependencies, verifies hashes and records what was left out', async () => {
    await withRoot(async root => {
      const gltf = Buffer.from(JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'mesh.bin', byteLength: 4 }],
        images: [{ uri: 'textures/a%20b.png' }, { uri: 'data:image/png;base64,AAAA' }] }));
      const bin = Buffer.from([1, 2, 3, 4]), png = Buffer.alloc(4096, 7), license = Buffer.from('credits\n');
      await writeFile(join(root, 'scene.zip'), zip([
        { name: 'pack/', data: Buffer.alloc(0) },
        { name: 'pack/scene.gltf', data: gltf, deflate: true },
        { name: 'pack/mesh.bin', data: bin },
        { name: 'pack/textures/a b.png', data: png, deflate: true },
        { name: 'pack/credits.txt', data: license },
        { name: 'pack/unused.txt', data: Buffer.from('not selected') },
      ]));
      const archiveSha = sha256(await readFile(join(root, 'scene.zip')));
      run(['--archive', join(root, 'scene.zip'), '--output', join(root, 'out'), '--archive-sha256', archiveSha,
        '--gltf', 'pack/scene.gltf', '--entry', 'pack/credits.txt', '--expect', `pack/mesh.bin=${sha256(bin)}`, '--strip-prefix', 'pack/']);
      const manifest = JSON.parse(await readFile(join(root, 'out', 'manifest.json'), 'utf8'));
      expect(manifest.archive).toMatchObject({ sha256: archiveSha, expectedSha256: archiveSha, entries: 6 });
      expect(manifest.files.map((file: { file: string }) => file.file)).toEqual(['credits.txt', 'mesh.bin', 'scene.gltf', 'textures/a b.png']);
      for (const file of manifest.files) {
        const bytes = await readFile(join(root, 'out', ...file.file.split('/')));
        expect(sha256(bytes)).toBe(file.sha256);
        expect(file.bytes).toBe(bytes.length);
      }
      expect(manifest.files.find((file: { file: string }) => file.file === 'mesh.bin').expectedSha256).toBe(sha256(bin));
      expect(manifest.notExtracted).toEqual([{ entry: 'pack/unused.txt', bytes: 12, crc32: crc32(Buffer.from('not selected')).toString(16).padStart(8, '0') }]);
      expect(() => run(['--archive', join(root, 'scene.zip'), '--output', join(root, 'out'), '--entry', 'pack/mesh.bin'])).toThrow(/EXTRACT_OUTPUT_EXISTS/);
    });
  });

  it('fails without publishing on hash, CRC or name violations', async () => {
    await withRoot(async root => {
      await writeFile(join(root, 'bad.zip'), zip([
        { name: 'ok.bin', data: Buffer.from('fine') },
        { name: 'broken.bin', data: Buffer.alloc(64, 3), corrupt: true },
        { name: '../escape.txt', data: Buffer.from('x') },
      ]));
      const archive = join(root, 'bad.zip');
      expect(() => run(['--archive', archive, '--output', join(root, 'a'), '--archive-sha256', '0'.repeat(64), '--entry', 'ok.bin'])).toThrow(/EXTRACT_ARCHIVE_SHA256_MISMATCH/);
      expect(() => run(['--archive', archive, '--output', join(root, 'b'), '--entry', 'broken.bin'])).toThrow(/EXTRACT_CRC_OR_LENGTH_MISMATCH/);
      expect(() => run(['--archive', archive, '--output', join(root, 'c'), '--entry', '../escape.txt'])).toThrow(/EXTRACT_UNSAFE_ENTRY_NAME/);
      expect(() => run(['--archive', archive, '--output', join(root, 'd'), '--entry', 'ok.bin', '--expect', `ok.bin=${'1'.repeat(64)}`])).toThrow(/EXTRACT_ENTRY_SHA256_MISMATCH/);
      expect(() => run(['--archive', archive, '--output', join(root, 'e'), '--entry', 'missing.bin'])).toThrow(/EXTRACT_MISSING_ENTRY/);
      expect((await readdir(root)).sort()).toEqual(['bad.zip']);
    });
  });
});
