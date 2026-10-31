// Shared texture package builder used by build-texture.mjs (one image) and
// build-gltf-textures.mjs (a whole scene). Two encoder paths:
//  - 'texconv': external Microsoft DirectXTex texconv (Windows), unchanged;
//  - 'builtin': cross-platform PNG/TGA/legacy-DDS decode, the engine's
//    buildMipChain semantics and the repo's deterministic CPU BC7 encoder on
//    worker threads. Legacy DDS sources keep only their top level.
// Both publish the same layout: manifest.json, build.json, source.<ext> (a
// re-encoded DDS source is kept as source-input.dds), source.dds and
// mips/<level>.bin. DDS inputs without an encoder are packaged as-is.
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { open, realpath, mkdir, writeFile, readFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import zlib from 'node:zlib';
import { createServer } from 'vite';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
export const LIMITS = Object.freeze({ inputBytes: 384 * MiB, pngBytes: 128 * MiB,
  outputBytes: 768 * MiB, executableBytes: 32 * MiB, dimension: 8192 });
const execute = promisify(execFile);
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
export const BUILTIN_DEFAULTS = Object.freeze({ bc7Quality: 2, bc7EarlyOutError: 96 });
const CODEC_MODULES = ['src/texture-bc7.ts', 'src/texture-mip-chain.ts', 'src/texture-image-decode.ts', 'src/texture-bcn.ts',
  'scripts/texture-codec-pool.mjs', 'scripts/texture-codec-worker.mjs', 'scripts/texture-package.mjs'];

export async function boundedRead(path, limit) {
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

function mipBytes(width, height, format) {
  let w = width, h = height, bytes = 0;
  for (;;) {
    bytes += format === 'bc7' ? Math.ceil(w / 4) * Math.ceil(h / 4) * 16 : w * h * 4;
    if (w === 1 && h === 1) return bytes;
    w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2));
  }
}

function pngLayout(bytes, format) {
  if (bytes.length < 33 || !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    || bytes.readUInt32BE(8) !== 13 || bytes.toString('ascii', 12, 16) !== 'IHDR') throw Error('TEXTURE_BUILD_PNG_HEADER');
  const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20);
  if (width < 1 || height < 1 || width > LIMITS.dimension || height > LIMITS.dimension
    || (format === 'bc7' && (width % 4 || height % 4))) throw Error('TEXTURE_BUILD_PNG_DIMENSIONS');
  // The external codec performs complete PNG decoding. Header bounds are checked
  // first, before permitting it to allocate the decoded image or mip chain.
  const nativeBytes = mipBytes(width, height, format);
  if (bytes.length + nativeBytes * 2 + 148 + MiB > LIMITS.outputBytes) throw Error('TEXTURE_BUILD_OUTPUT_BYTE_BUDGET');
  return { width, height, nativeBytes };
}

/** DX10 DDS container for already encoded levels (parsed back by parseTextureDds). */
export function ddsContainer(width, height, format, colorSpace, levels) {
  const header = Buffer.alloc(148);
  const compressed = format === 'bc7';
  header.write('DDS ', 0, 'ascii');
  header.writeUInt32LE(124, 4);
  header.writeUInt32LE(0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | (compressed ? 0x80000 : 0x8), 8);
  header.writeUInt32LE(height, 12);
  header.writeUInt32LE(width, 16);
  header.writeUInt32LE(compressed ? Math.ceil(width / 4) * Math.ceil(height / 4) * 16 : width * 4, 20);
  header.writeUInt32LE(levels.length, 28);
  header.writeUInt32LE(32, 76);
  header.writeUInt32LE(0x4, 80);
  header.write('DX10', 84, 'ascii');
  header.writeUInt32LE(0x401008, 108);
  header.writeUInt32LE(compressed ? (colorSpace === 'srgb' ? 99 : 98) : (colorSpace === 'srgb' ? 29 : 28), 128);
  header.writeUInt32LE(3, 132);
  header.writeUInt32LE(0, 136);
  header.writeUInt32LE(1, 140);
  header.writeUInt32LE(0, 144);
  return Buffer.concat([header, ...levels.map(level => Buffer.from(level.buffer, level.byteOffset, level.byteLength))]);
}

/**
 * Per-level error of a decoded level against its RGBA8 reference (stored-byte
 * space). `weights` marks the channels the material actually reads; the
 * `*Used` fields cover only those, the other fields always cover all channels.
 */
export function levelError(reference, decoded, weights = [1, 1, 1, 1]) {
  const se = [0, 0, 0, 0], max = [0, 0, 0, 0];
  for (let i = 0; i < reference.length; i++) {
    const d = decoded[i] - reference[i], c = i & 3;
    se[c] += d * d;
    if (d > max[c] || -d > max[c]) max[c] = Math.abs(d);
  }
  const pixels = reference.length / 4, rgb = (se[0] + se[1] + se[2]) / (3 * pixels);
  const psnr = mse => mse === 0 ? null : 10 * Math.log10(255 * 255 / mse);
  const used = [0, 1, 2, 3].filter(c => weights[c] > 0);
  const usedMse = used.reduce((sum, c) => sum + se[c], 0) / (used.length * pixels);
  return { pixels, sse: se, maxAbs: max, rmse: se.map(v => Math.sqrt(v / pixels) / 255),
    rmseRgb: Math.sqrt(rgb) / 255, psnrRgb: psnr(rgb), psnrAlpha: psnr(se[3] / pixels),
    usedChannels: used, psnrUsed: psnr(usedMse), maxAbsUsed: Math.max(...used.map(c => max[c])) };
}

async function moduleHashes() {
  const result = {};
  for (const file of CODEC_MODULES) result[file] = sha256(await readFile(join(ROOT, file)));
  return result;
}

export async function createTextureContext() {
  const server = await createServer({ root: ROOT, configFile: false,
    server: { middlewareMode: true, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
  const modules = {
    ...await server.ssrLoadModule('/src/texture-dds.ts'),
    ...await server.ssrLoadModule('/src/texture-stream-manifest.ts'),
    ...await server.ssrLoadModule('/src/texture-image-decode.ts'),
    ...await server.ssrLoadModule('/src/texture-mip-chain.ts'),
    ...await server.ssrLoadModule('/src/texture-bc7.ts'),
    ...await server.ssrLoadModule('/src/texture-bcn.ts'),
  };
  return { modules, hashes: await moduleHashes(), close: () => server.close() };
}

function decodeSource(config, input, modules) {
  const { decodePng, decodeTga, decodeLegacyDds } = modules;
  const bytes = new Uint8Array(input.bytes.buffer, input.bytes.byteOffset, input.bytes.byteLength);
  if (config.extension === '.png') return decodePng(bytes, data => zlib.inflateSync(data, { maxOutputLength: 8192 * 8192 * 8 + 8192 }));
  if (config.extension === '.tga') return decodeTga(bytes);
  const dds = decodeLegacyDds(bytes, LIMITS.dimension);
  return { width: dds.width, height: dds.height, pixels: dds.pixels,
    source: { format: 'dds', fourCC: dds.fourCC, encoding: dds.format, sourceLevels: dds.mipCount, channels: dds.channels, decodedLevels: 1 } };
}

async function builtinEncode(config, input, context, pool) {
  const { buildRgba8MipChain, decodeBc7Image, encodeBc7Image, bc7ModeHistogram, convertNormalMap } = context.modules;
  const started = Date.now();
  const decoded = decodeSource(config, input, context.modules);
  if (config.format === 'bc7' && (decoded.width % 4 || decoded.height % 4)) throw Error('TEXTURE_BUILD_BC7_DIMENSIONS');
  // Tangent-space normal policy is explicit: the DirectX green axis is flipped
  // to glTF's, and a two-channel source (BC5/ATI2) gets Z = sqrt(1 - x^2 - y^2).
  const decodedSha256 = sha256(decoded.pixels);
  const normalMap = config.normalConvention ? { convention: config.normalConvention, flippedGreen: config.normalConvention === 'directx',
    reconstructedZ: decoded.source.format === 'dds' && decoded.source.channels === 2 } : null;
  if (normalMap && (normalMap.flippedGreen || normalMap.reconstructedZ)) {
    convertNormalMap(decoded.pixels, { flipGreen: normalMap.flippedGreen, reconstructZ: normalMap.reconstructedZ });
    normalMap.convertedRgba8Sha256 = sha256(decoded.pixels);
  }
  const nativeBytes = mipBytes(decoded.width, decoded.height, config.format);
  if (input.bytes.length + nativeBytes * 2 + 148 + MiB > LIMITS.outputBytes) throw Error('TEXTURE_BUILD_OUTPUT_BYTE_BUDGET');
  const decodeMs = Date.now() - started;
  const chain = buildRgba8MipChain(decoded.width, decoded.height, decoded.pixels, config.colorSpace);
  const mipMs = Date.now() - started - decodeMs;
  const weights = config.bc7ChannelWeights ?? [1, 1, 1, 1];
  const options = { quality: config.bc7Quality, earlyOutError: config.bc7EarlyOutError, channelWeights: weights };
  const levels = [], quality = [];
  for (const level of chain) {
    if (config.format === 'rgba8') { levels.push(level.pixels); continue; }
    const blocks = pool && level.width * level.height >= 64 * 64
      ? await pool.encodeBc7(level.pixels, level.width, level.height, options)
      : encodeBc7Image(level.pixels, level.width, level.height, options);
    levels.push(blocks);
    quality.push({ level: quality.length, width: level.width, height: level.height, modes: bc7ModeHistogram(blocks),
      ...levelError(level.pixels, decodeBc7Image(blocks, level.width, level.height), weights) });
  }
  const encodeMs = Date.now() - started - decodeMs - mipMs;
  const encoded = ddsContainer(decoded.width, decoded.height, config.format, config.colorSpace, levels);
  const summary = quality.length ? (() => {
    const totals = quality.reduce((sum, level) => ({ pixels: sum.pixels + level.pixels, sse: sum.sse.map((v, c) => v + level.sse[c]),
      maxAbs: sum.maxAbs.map((v, c) => Math.max(v, level.maxAbs[c])) }), { pixels: 0, sse: [0, 0, 0, 0], maxAbs: [0, 0, 0, 0] });
    const rgb = (totals.sse[0] + totals.sse[1] + totals.sse[2]) / (3 * totals.pixels);
    const used = quality[0].usedChannels, usedMse = used.reduce((sum, c) => sum + totals.sse[c], 0) / (used.length * totals.pixels);
    return { top: quality[0], allLevels: { pixels: totals.pixels, maxAbs: totals.maxAbs, rmseRgb: Math.sqrt(rgb) / 255,
      psnrRgb: rgb ? 10 * Math.log10(255 * 255 / rgb) : null, psnrAlpha: totals.sse[3] ? 10 * Math.log10(255 * 255 * totals.pixels / totals.sse[3]) : null,
      usedChannels: used, psnrUsed: usedMse ? 10 * Math.log10(255 * 255 / usedMse) : null, maxAbsUsed: Math.max(...used.map(c => totals.maxAbs[c])) } };
  })() : null;
  const codec = { name: 'real-engine builtin', node: process.version, moduleSha256: context.hashes,
    decodedSource: { ...decoded.source, width: decoded.width, height: decoded.height, rgba8Sha256: decodedSha256 }, normalMap,
    mipFilter: { kernel: 'source-area box (buildMipChain semantics)', rgb: config.colorSpace === 'srgb' ? 'linear-light-srgb' : 'linear', alpha: 'independent', rounding: 'nearest' },
    encoder: config.format === 'bc7' ? { format: 'bc7', ...options, deterministic: 'independent of worker count and task order',
      channelWeightsMeaning: 'squared-error weight per RGBA channel; 0 marks a channel the material never reads',
      opaqueAlpha: 'blocks with all alpha 255 decode to exactly 255', errorSpace: config.colorSpace === 'srgb' ? 'stored sRGB bytes' : 'stored linear bytes' } : { format: 'rgba8', lossless: true },
    quality: summary ? { reference: 'buildMipChain RGBA8 level of the decoded source', summary, levels: quality } : null };
  // Timings and worker count are run diagnostics, kept out of build.json so the
  // package bytes are a pure function of the source and options.
  return { encoded, codec, run: { workers: pool?.size ?? 0, timingsMs: { decode: decodeMs, mips: mipMs, encodeAndVerify: encodeMs } } };
}

/** Builds one package into an empty private staging directory. */
export async function buildTextureInto(config, temporary, context, pool) {
  const { parseTextureDds, validateTextureStreamManifest } = context.modules;
  const input = await boundedRead(config.input, config.extension === '.dds' ? LIMITS.inputBytes : LIMITS.pngBytes);
  const reencodedDds = config.extension === '.dds' && config.encoder === 'builtin';
  const sourceFile = reencodedDds ? 'source-input.dds' : 'source' + config.extension;
  let codec = null, encoded = input.bytes, expectedPng, run = null;
  await writeFile(join(temporary, sourceFile), input.bytes, { flag: 'wx' });
  if (config.encoder === 'builtin') {
    ({ encoded, codec, run } = await builtinEncode(config, input, context, pool));
    await writeFile(join(temporary, 'source.dds'), encoded, { flag: 'wx' });
  } else if (config.extension === '.png') {
    expectedPng = pngLayout(input.bytes, config.format);
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
  const totalBytes = nativeBytes + input.bytes.length + (sourceFile !== 'source.dds' ? encoded.length : 0) + manifestBytes.length + buildBytes.length;
  if (totalBytes > LIMITS.outputBytes) throw Error('TEXTURE_BUILD_OUTPUT_BYTE_BUDGET');
  await mkdir(join(temporary, 'mips'));
  for (let mip = 0; mip < mips.length; mip++) await writeFile(join(temporary, mips[mip].file), asset.levels[mip].data, { flag: 'wx' });
  await writeFile(join(temporary, 'build.json'), buildBytes, { flag: 'wx' });
  await writeFile(join(temporary, 'manifest.json'), manifestBytes, { flag: 'wx' });
  return { id: manifest.id, width: asset.width, height: asset.height, format: asset.format, colorSpace: asset.colorSpace,
    levels: mips.length, fallbackMip, nativeBytes, fallbackBytes: build.fallbackBytes, outputBytes: totalBytes,
    quality: codec?.quality?.summary ?? null, run };
}
