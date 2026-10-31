#!/usr/bin/env node
// Verified, selective ZIP extraction for large third-party scene archives.
// Pure Node (zlib inflateRaw + CRC32 + SHA-256), so the same command works on
// macOS, Linux and Windows without external unzip versions. The archive is
// untrusted data: entry names are sanitized, nothing is executed, encrypted or
// unsupported entries are rejected, and the output directory must not exist.
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, realpathSync } from 'node:fs';
import { open, mkdir, mkdtemp, writeFile, readFile } from 'node:fs/promises';
import { basename, dirname, join, posix, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Transform } from 'node:stream';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ entries: 65536, entryBytes: 8 * 1024 * MiB, totalBytes: 64 * 1024 * MiB, centralDirectoryBytes: 64 * MiB });

function usage() {
  return 'usage: extract-archive --archive file.zip --output new-directory [--archive-sha256 hex] '
    + '[--entry path]... [--gltf path-in-archive]... [--prefix path/]... [--expect path=sha256]... [--strip-prefix dir/]';
}

export function parseArguments(argv) {
  const options = { entries: [], gltf: [], prefixes: [], expect: new Map(), strip: '' };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i], value = argv[i + 1];
    if (!value || value.startsWith('--')) throw Error('EXTRACT_ARGUMENTS: ' + key + '\n' + usage());
    i++;
    if (key === '--archive') options.archive = resolve(value);
    else if (key === '--output') options.output = resolve(value);
    else if (key === '--archive-sha256') {
      if (!/^[\da-f]{64}$/i.test(value)) throw Error('EXTRACT_ARCHIVE_SHA256');
      options.archiveSha256 = value.toLowerCase();
    } else if (key === '--entry') options.entries.push(value);
    else if (key === '--gltf') options.gltf.push(value);
    else if (key === '--prefix') options.prefixes.push(value);
    else if (key === '--strip-prefix') options.strip = value.endsWith('/') ? value : value + '/';
    else if (key === '--expect') {
      const at = value.lastIndexOf('=');
      if (at < 1 || !/^[\da-f]{64}$/i.test(value.slice(at + 1))) throw Error('EXTRACT_EXPECT: ' + value);
      options.expect.set(value.slice(0, at), value.slice(at + 1).toLowerCase());
    } else throw Error('EXTRACT_ARGUMENTS: ' + key + '\n' + usage());
  }
  if (!options.archive || !options.output) throw Error(usage());
  if (!options.entries.length && !options.gltf.length && !options.prefixes.length) throw Error('EXTRACT_NOTHING_SELECTED\n' + usage());
  return options;
}

/** Rejects absolute, parent-relative, drive-qualified and control-character names. */
export function safeEntryName(name) {
  if (typeof name !== 'string' || !name.length || name.length > 1024 || /[\u0000-\u001f\u007f\\]/.test(name)
    || name.startsWith('/') || /^[A-Za-z]:/.test(name)) return false;
  return name.split('/').every((segment, index, all) => segment !== '..' && segment !== '.'
    && (segment.length > 0 || index === all.length - 1));
}

async function readAt(handle, position, length) {
  const buffer = Buffer.alloc(length);
  let offset = 0;
  while (offset < length) {
    const { bytesRead } = await handle.read(buffer, offset, length - offset, position + offset);
    if (!bytesRead) throw Error('EXTRACT_TRUNCATED_ARCHIVE');
    offset += bytesRead;
  }
  return buffer;
}

/** Parses the central directory, including ZIP64 records and extra fields. */
export async function readCentralDirectory(handle, size) {
  const tailLength = Math.min(size, 65557);
  const tail = await readAt(handle, size - tailLength, tailLength);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) if (tail.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw Error('EXTRACT_NO_END_OF_CENTRAL_DIRECTORY');
  let count = tail.readUInt16LE(eocd + 10), cdSize = tail.readUInt32LE(eocd + 12), cdOffset = tail.readUInt32LE(eocd + 16);
  if (eocd >= 20 && tail.readUInt32LE(eocd - 20) === 0x07064b50) {
    const recordOffset = Number(tail.readBigUInt64LE(eocd - 12));
    const record = await readAt(handle, recordOffset, 56);
    if (record.readUInt32LE(0) !== 0x06064b50) throw Error('EXTRACT_ZIP64_RECORD');
    count = Number(record.readBigUInt64LE(32)); cdSize = Number(record.readBigUInt64LE(40)); cdOffset = Number(record.readBigUInt64LE(48));
  }
  if (count > LIMITS.entries || cdSize > LIMITS.centralDirectoryBytes || cdOffset + cdSize > size) throw Error('EXTRACT_CENTRAL_DIRECTORY_BOUNDS');
  const cd = await readAt(handle, cdOffset, cdSize), entries = [];
  for (let p = 0, n = 0; n < count; n++) {
    if (p + 46 > cd.length || cd.readUInt32LE(p) !== 0x02014b50) throw Error('EXTRACT_CENTRAL_DIRECTORY_ENTRY');
    const flags = cd.readUInt16LE(p + 8), method = cd.readUInt16LE(p + 10), crc32 = cd.readUInt32LE(p + 16);
    let compressedSize = cd.readUInt32LE(p + 20), size = cd.readUInt32LE(p + 24);
    const nameLength = cd.readUInt16LE(p + 28), extraLength = cd.readUInt16LE(p + 30), commentLength = cd.readUInt16LE(p + 32);
    let localOffset = cd.readUInt32LE(p + 42);
    const name = cd.toString(flags & 0x800 ? 'utf8' : 'latin1', p + 46, p + 46 + nameLength);
    const extra = cd.subarray(p + 46 + nameLength, p + 46 + nameLength + extraLength);
    for (let e = 0; e + 4 <= extra.length;) {
      const id = extra.readUInt16LE(e), length = extra.readUInt16LE(e + 2);
      if (id === 0x0001) {
        let q = e + 4;
        if (size === 0xffffffff) { size = Number(extra.readBigUInt64LE(q)); q += 8; }
        if (compressedSize === 0xffffffff) { compressedSize = Number(extra.readBigUInt64LE(q)); q += 8; }
        if (localOffset === 0xffffffff) { localOffset = Number(extra.readBigUInt64LE(q)); q += 8; }
      }
      e += 4 + length;
    }
    entries.push({ name, flags, method, crc32, compressedSize, size, localOffset, directory: name.endsWith('/') });
    p += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

async function hashFile(path) {
  const hash = createHash('sha256');
  await pipeline(createReadStream(path, { highWaterMark: 4 * MiB }), new Transform({ transform(chunk, _, done) { hash.update(chunk); done(); } }));
  return hash.digest('hex');
}

async function extractEntry(handle, archive, entry, destination) {
  if (entry.flags & 0x1) throw Error('EXTRACT_ENCRYPTED_ENTRY: ' + entry.name);
  if (entry.method !== 0 && entry.method !== 8) throw Error('EXTRACT_UNSUPPORTED_METHOD: ' + entry.name);
  if (entry.size > LIMITS.entryBytes) throw Error('EXTRACT_ENTRY_BYTE_BUDGET: ' + entry.name);
  const local = await readAt(handle, entry.localOffset, 30);
  if (local.readUInt32LE(0) !== 0x04034b50) throw Error('EXTRACT_LOCAL_HEADER: ' + entry.name);
  const start = entry.localOffset + 30 + local.readUInt16LE(26) + local.readUInt16LE(28);
  const hash = createHash('sha256');
  let crc = 0, written = 0;
  const verify = new Transform({ transform(chunk, _, done) {
    written += chunk.length;
    if (written > entry.size) { done(Error('EXTRACT_ENTRY_LONGER_THAN_DECLARED: ' + entry.name)); return; }
    crc = zlib.crc32(chunk, crc); hash.update(chunk); done(null, chunk);
  } });
  await mkdir(dirname(destination), { recursive: true });
  const source = entry.compressedSize ? createReadStream(archive, { start, end: start + entry.compressedSize - 1, highWaterMark: 4 * MiB }) : null;
  const stages = entry.method === 8 ? [zlib.createInflateRaw({ chunkSize: 1 * MiB })] : [];
  if (source) await pipeline(source, ...stages, verify, createWriteStream(destination, { flags: 'wx' }));
  else await writeFile(destination, Buffer.alloc(0), { flag: 'wx' });
  if (written !== entry.size || (crc >>> 0) !== entry.crc32) throw Error('EXTRACT_CRC_OR_LENGTH_MISMATCH: ' + entry.name);
  return { bytes: written, crc32: (crc >>> 0).toString(16).padStart(8, '0'), sha256: hash.digest('hex') };
}

function gltfDependencies(gltfName, json) {
  const base = posix.dirname(gltfName), names = [];
  for (const item of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    if (typeof item.uri !== 'string' || item.uri.startsWith('data:')) continue;
    if (/^[a-z][a-z0-9+.-]*:/i.test(item.uri)) throw Error('EXTRACT_GLTF_REMOTE_URI: ' + item.uri);
    names.push(posix.normalize(posix.join(base, decodeURIComponent(item.uri))));
  }
  return names;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (await exists(options.output)) throw Error('EXTRACT_OUTPUT_EXISTS: ' + options.output);
  const started = Date.now();
  const handle = await open(options.archive, 'r');
  const parent = dirname(options.output);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, `.${basename(options.output)}.tmp-`));
  try {
    const info = await handle.stat();
    const archiveSha256 = await hashFile(options.archive);
    if (options.archiveSha256 && archiveSha256 !== options.archiveSha256) throw Error('EXTRACT_ARCHIVE_SHA256_MISMATCH: ' + archiveSha256);
    const entries = await readCentralDirectory(handle, info.size);
    const byName = new Map(entries.map(entry => [entry.name, entry]));
    const selected = new Set(options.entries);
    for (const prefix of options.prefixes) for (const entry of entries) if (!entry.directory && entry.name.startsWith(prefix)) selected.add(entry.name);
    const files = [];
    const take = async name => {
      const entry = byName.get(name);
      if (!entry || entry.directory) throw Error('EXTRACT_MISSING_ENTRY: ' + name);
      if (!safeEntryName(name) || (options.strip && !name.startsWith(options.strip))) throw Error('EXTRACT_UNSAFE_ENTRY_NAME: ' + name);
      const relativeName = name.slice(options.strip.length);
      if (!safeEntryName(relativeName)) throw Error('EXTRACT_UNSAFE_ENTRY_NAME: ' + name);
      const result = await extractEntry(handle, options.archive, entry, join(temporary, ...relativeName.split('/')));
      const expected = options.expect.get(name);
      if (expected && expected !== result.sha256) throw Error(`EXTRACT_ENTRY_SHA256_MISMATCH: ${name} ${result.sha256}`);
      files.push({ entry: name, file: relativeName, compressedBytes: entry.compressedSize, method: entry.method === 8 ? 'deflate' : 'stored', ...result,
        expectedSha256: expected ?? null });
      process.stdout.write(`extracted ${relativeName} ${result.bytes} bytes\n`);
      return relativeName;
    };
    for (const gltf of options.gltf) {
      if (!selected.has(gltf)) selected.add(gltf);
    }
    const done = new Set();
    for (const name of [...selected].sort()) { await take(name); done.add(name); }
    for (const gltf of options.gltf) {
      const json = JSON.parse(await readFile(join(temporary, ...gltf.slice(options.strip.length).split('/')), 'utf8'));
      for (const dependency of gltfDependencies(gltf, json)) if (!done.has(dependency)) { await take(dependency); done.add(dependency); }
    }
    for (const [name] of options.expect) if (!done.has(name)) throw Error('EXTRACT_EXPECTED_ENTRY_NOT_SELECTED: ' + name);
    files.sort((a, b) => a.file < b.file ? -1 : a.file > b.file ? 1 : 0);
    const manifest = { version: 1, format: 'real-engine-archive-extraction', tool: 'scripts/extract-archive.mjs', node: process.version,
      archive: { file: basename(options.archive), bytes: info.size, sha256: archiveSha256, expectedSha256: options.archiveSha256 ?? null,
        entries: entries.length, declaredUncompressedBytes: entries.reduce((sum, entry) => sum + entry.size, 0) },
      selection: { entries: options.entries, gltf: options.gltf, prefixes: options.prefixes, stripPrefix: options.strip || null },
      extractedFiles: files.length, extractedBytes: files.reduce((sum, file) => sum + file.bytes, 0), elapsedMs: Date.now() - started,
      files, notExtracted: entries.filter(entry => !entry.directory && !done.has(entry.name)).map(entry => ({ entry: entry.name, bytes: entry.size, crc32: entry.crc32.toString(16).padStart(8, '0') })) };
    if (manifest.extractedBytes > LIMITS.totalBytes) throw Error('EXTRACT_TOTAL_BYTE_BUDGET');
    await writeFile(join(temporary, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
    // Exclusive destination claim, hard-linked payloads, manifest linked last.
    await publishDirectory(temporary, options.output, { hardLinkFiles: true });
    process.stdout.write(JSON.stringify({ output: options.output, files: files.length, bytes: manifest.extractedBytes, archiveSha256 }) + '\n');
  } finally {
    await handle.close();
    await removeOwnTemporary(temporary, parent, options.output).catch(() => {});
  }
}

const invokedDirectly = () => { try { return realpathSync(process.argv[1]) === fileURLToPath(import.meta.url); } catch { return false; } };
if (process.argv[1] && invokedDirectly()) {
  main().catch(error => { process.stderr.write(`extract-archive: ${error.message}\n`); process.exitCode = 1; });
}
