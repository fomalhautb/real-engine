#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { open, realpath, mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { createServer } from 'vite';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ inputBytes: 384 * MiB, pngBytes: 128 * MiB,
  outputBytes: 768 * MiB, executableBytes: 32 * MiB, dimension: 8192 });
const execute = promisify(execFile);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');

function argumentsOf(argv) {
  const values = new Map();
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!['--input', '--output', '--id', '--fallback-size', '--format', '--color-space', '--texconv', '--gpu-adapter'].includes(key)
      || values.has(key) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw Error('TEXTURE_BUILD_ARGUMENTS');
    values.set(key, argv[++i]);
  }
  if (!values.has('--input') || !values.has('--output')) throw Error('usage: build-texture --input image.dds|png --output new-directory [--id id] [--fallback-size 64] [--format rgba8|bc7 --color-space srgb|linear --texconv path] [--gpu-adapter index]');
  const input = resolve(values.get('--input')), output = resolve(values.get('--output'));
  const extension = extname(input).toLowerCase();
  if (!['.dds', '.png'].includes(extension) || input === output) throw Error('TEXTURE_BUILD_INPUT');
  const id = values.get('--id') ?? basename(input, extname(input));
  if (id.trim().length < 1 || id.length > 128 || /[\u0000-\u001f\u007f]/.test(id)) throw Error('TEXTURE_BUILD_ID');
  const fallbackSize = Number(values.get('--fallback-size') ?? 64);
  if (!Number.isSafeInteger(fallbackSize) || fallbackSize < 1 || fallbackSize > LIMITS.dimension) throw Error('TEXTURE_BUILD_FALLBACK_SIZE');
  const format = values.get('--format'), colorSpace = values.get('--color-space');
  const texconv = values.has('--texconv') ? resolve(values.get('--texconv')) : undefined;
  if (format !== undefined && !['rgba8', 'bc7'].includes(format)) throw Error('TEXTURE_BUILD_FORMAT');
  if (colorSpace !== undefined && !['srgb', 'linear'].includes(colorSpace)) throw Error('TEXTURE_BUILD_COLOR_SPACE');
  if (extension === '.png' && (!format || !colorSpace || !texconv)) throw Error('TEXTURE_BUILD_PNG_REQUIRES_FORMAT_COLOR_SPACE_TEXCONV');
  if (extension === '.dds' && texconv) throw Error('TEXTURE_BUILD_DDS_DOES_NOT_RUN_CODEC');
  const gpuAdapter = values.has('--gpu-adapter') ? Number(values.get('--gpu-adapter')) : undefined;
  if (gpuAdapter !== undefined && (!Number.isSafeInteger(gpuAdapter) || gpuAdapter < 0 || gpuAdapter > 15
    || extension !== '.png' || format !== 'bc7')) throw Error('TEXTURE_BUILD_GPU_ADAPTER');
  return { input, output, id, fallbackSize, extension, format, colorSpace, texconv, gpuAdapter };
}

async function boundedRead(path, limit) {
  const canonical = await realpath(path), handle = await open(canonical, 'r');
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size < 1 || before.size > limit) throw Error('TEXTURE_BUILD_INPUT_BYTE_BUDGET');
    const bytes = Buffer.alloc(before.size);
    for (let offset = 0; offset < bytes.length;) {
      const read = await handle.read(bytes, offset, Math.min(4 * MiB, bytes.length - offset), offset);
      if (!read.bytesRead) throw Error('TEXTURE_BUILD_SOURCE_CHANGED');
      offset += read.bytesRead;
    }
    const extra = await handle.read(Buffer.alloc(1), 0, 1, bytes.length), after = await handle.stat();
    if (extra.bytesRead || before.size !== after.size || before.mtimeMs !== after.mtimeMs || await realpath(path) !== canonical) throw Error('TEXTURE_BUILD_SOURCE_CHANGED');
    return { bytes, canonical };
  } finally { await handle.close(); }
}

function pngLayout(bytes, format) {
  if (bytes.length < 33 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    || bytes.readUInt32BE(8) !== 13 || bytes.toString('ascii', 12, 16) !== 'IHDR') throw Error('TEXTURE_BUILD_PNG_HEADER');
  const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20);
  if (width < 1 || height < 1 || width > LIMITS.dimension || height > LIMITS.dimension
    || (format === 'bc7' && (width % 4 || height % 4))) throw Error('TEXTURE_BUILD_PNG_DIMENSIONS');
  // The external codec performs complete PNG decoding. Header bounds are checked
  // first, before permitting it to allocate the decoded image or mip chain.
  let w = width, h = height, nativeBytes = 0;
  for (;;) {
    nativeBytes += format === 'bc7' ? Math.ceil(w / 4) * Math.ceil(h / 4) * 16 : w * h * 4;
    if (w === 1 && h === 1) break;
    w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2));
  }
  if (bytes.length + nativeBytes * 2 + 148 + MiB > LIMITS.outputBytes) throw Error('TEXTURE_BUILD_OUTPUT_BYTE_BUDGET');
  return { width, height, nativeBytes };
}

async function buildInto(config, temporary) {
  const server = await createServer({ root: ROOT, configFile: false,
    server: { middlewareMode: true, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
  try {
    const { parseTextureDds } = await server.ssrLoadModule('/src/texture-dds.ts');
    const { validateTextureStreamManifest } = await server.ssrLoadModule('/src/texture-stream-manifest.ts');
    const input = await boundedRead(config.input, config.extension === '.png' ? LIMITS.pngBytes : LIMITS.inputBytes);
    const sourceFile = 'source' + config.extension;
    let codec = null, encoded = input.bytes, expectedPng;
    if (config.extension === '.png') expectedPng = pngLayout(input.bytes, config.format);
    await writeFile(join(temporary, sourceFile), input.bytes, { flag: 'wx' });
    if (expectedPng) {
      const tool = await boundedRead(config.texconv, LIMITS.executableBytes);
      const dxgi = (config.format === 'bc7' ? 'BC7_UNORM' : 'R8G8B8A8_UNORM') + (config.colorSpace === 'srgb' ? '_SRGB' : '');
      // Independent channels match buildMipChain: linear-light RGB for sRGB,
      // linear alpha/data. No resize, Y flip, normal reconstruction, or repair.
      const args = ['-nologo', '-dx10', '-m', '0', '-if', 'BOX', '-nowic', '-sepalpha', '--ignore-srgb', '-f', dxgi];
      if (config.gpuAdapter === undefined) args.push('-nogpu');
      else args.push('-gpu', String(config.gpuAdapter));
      if (config.colorSpace === 'srgb') args.push('-srgb');
      args.push('-o', '.', '--', sourceFile);
      const env = { ...process.env, OMP_NUM_THREADS: '4' };
      const version = await execute(tool.canonical, ['--version'], { windowsHide: true, timeout: 10000, maxBuffer: 16384, env });
      let result;
      try {
        result = await execute(tool.canonical, args, { cwd: temporary, windowsHide: true, timeout: 600000, maxBuffer: MiB, env });
      } catch (error) {
        throw Error(`TEXTURE_BUILD_CODEC_FAILED: code=${error.code ?? 'none'} signal=${error.signal ?? 'none'} killed=${Boolean(error.killed)}; ${String(error.stderr ?? '').slice(0, 4096)}`);
      }
      encoded = (await boundedRead(join(temporary, 'source.dds'), LIMITS.inputBytes)).bytes;
      codec = { name: 'Microsoft DirectXTex texconv', version: version.stdout.trim(),
        executableSha256: sha256(tool.bytes), args, cpuThreadsRequested: 4,
        gpuAdapterRequested: config.gpuAdapter ?? null, output: result.stdout.trim(), stderr: result.stderr.trim(),
        mipFilter: { kernel: 'box', rgb: config.colorSpace === 'srgb' ? 'linear-light-srgb' : 'linear', alpha: 'independent' } };
    }
    const asset = parseTextureDds(new Uint8Array(encoded.buffer, encoded.byteOffset, encoded.byteLength));
    if ((config.format && asset.format !== config.format) || (config.colorSpace && asset.colorSpace !== config.colorSpace)
      || (expectedPng && (asset.width !== expectedPng.width || asset.height !== expectedPng.height))) throw Error('TEXTURE_BUILD_ENCODED_LAYOUT');
    let fallbackMip = -1, coarsestAdmissible = -1;
    for (let mip = 0; mip < asset.levels.length; mip++) {
      const level = asset.levels[mip];
      if (asset.format !== 'rgba8' && (level.width % 4 || level.height % 4)) continue;
      coarsestAdmissible = mip;
      if (fallbackMip < 0 && Math.max(level.width, level.height) <= config.fallbackSize) fallbackMip = mip;
    }
    if (fallbackMip < 0) fallbackMip = coarsestAdmissible;
    if (fallbackMip < 0) throw Error('TEXTURE_BUILD_FALLBACK_LAYOUT');
    const mips = asset.levels.map((level, index) => ({ level: index, width: level.width, height: level.height,
      byteLength: level.data.byteLength, sha256: sha256(level.data), file: `mips/${index}.bin` }));
    const manifest = validateTextureStreamManifest({ version: 1, id: config.id, width: asset.width, height: asset.height,
      format: asset.format, colorSpace: asset.colorSpace, fallbackMip, mips });
    const nativeBytes = mips.reduce((sum, level) => sum + level.byteLength, 0);
    const build = { version: 1, source: { file: sourceFile, byteLength: input.bytes.length, sha256: sha256(input.bytes) },
      encoded: { file: 'source.dds', byteLength: encoded.length, sha256: sha256(encoded) }, codec,
      nativeBytes, fallbackBytes: mips.slice(fallbackMip).reduce((sum, level) => sum + level.byteLength, 0),
      fallbackSize: config.fallbackSize, limits: LIMITS };
    const manifestBytes = json(manifest), buildBytes = json(build);
    const totalBytes = nativeBytes + input.bytes.length + (expectedPng ? encoded.length : 0) + manifestBytes.length + buildBytes.length;
    if (totalBytes > LIMITS.outputBytes) throw Error('TEXTURE_BUILD_OUTPUT_BYTE_BUDGET');
    await mkdir(join(temporary, 'mips'));
    for (let mip = 0; mip < mips.length; mip++) await writeFile(join(temporary, mips[mip].file), asset.levels[mip].data, { flag: 'wx' });
    await writeFile(join(temporary, 'build.json'), buildBytes, { flag: 'wx' });
    await writeFile(join(temporary, 'manifest.json'), manifestBytes, { flag: 'wx' });
    return { id: manifest.id, width: asset.width, height: asset.height, format: asset.format, colorSpace: asset.colorSpace,
      levels: mips.length, fallbackMip, nativeBytes, fallbackBytes: build.fallbackBytes, outputBytes: totalBytes };
  } finally { await server.close(); }
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) throw Error('TEXTURE_BUILD_OUTPUT_EXISTS');
  await mkdir(dirname(config.output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(config.output), '.' + basename(config.output) + '.tmp-'));
  try {
    const result = await buildInto(config, temporary);
    await publishDirectory(temporary, config.output);
    console.log(JSON.stringify(result));
  } finally { await removeOwnTemporary(temporary, dirname(config.output), config.output); }
}

main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
