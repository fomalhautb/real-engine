#!/usr/bin/env node
// One texture package. DDS inputs are packaged as-is (no codec) unless
// `--encoder builtin` re-encodes a legacy FourCC DDS (DXT1-5, ATI1/2). PNG/TGA
// inputs are encoded either by the external DirectXTex texconv (`--texconv path`)
// or by the repo's deterministic cross-platform encoder (`--encoder builtin`).
// `--normal-convention directx` marks a tangent-space normal map authored with
// the DirectX green axis (flipped to glTF); two-channel sources rebuild Z.
import { mkdir, mkdtemp } from 'node:fs/promises';
import { basename, dirname, extname, resolve, join } from 'node:path';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';
import { BUILTIN_DEFAULTS, LIMITS, buildTextureInto, createTextureContext } from './texture-package.mjs';
import { TextureCodecPool, defaultWorkerCount } from './texture-codec-pool.mjs';

const KEYS = ['--input', '--output', '--id', '--fallback-size', '--format', '--color-space', '--texconv', '--gpu-adapter',
  '--encoder', '--workers', '--bc7-quality', '--bc7-early-out', '--bc7-channel-weights', '--normal-convention'];
const USAGE = 'usage: build-texture --input image.dds|png|tga --output new-directory [--id id] [--fallback-size 64] '
  + '[--format rgba8|bc7 --color-space srgb|linear (--texconv path [--gpu-adapter index] | --encoder builtin [--workers n] [--bc7-quality 0|1|2] [--bc7-early-out sse] [--bc7-channel-weights r,g,b,a] [--normal-convention opengl|directx])]';

function argumentsOf(argv) {
  const values = new Map();
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!KEYS.includes(key) || values.has(key) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw Error('TEXTURE_BUILD_ARGUMENTS');
    values.set(key, argv[++i]);
  }
  if (!values.has('--input') || !values.has('--output')) throw Error(USAGE);
  const input = resolve(values.get('--input')), output = resolve(values.get('--output'));
  const extension = extname(input).toLowerCase();
  if (!['.dds', '.png', '.tga'].includes(extension) || input === output) throw Error('TEXTURE_BUILD_INPUT');
  const id = values.get('--id') ?? basename(input, extname(input));
  if (id.trim().length < 1 || id.length > 128 || /[\u0000-\u001f\u007f]/.test(id)) throw Error('TEXTURE_BUILD_ID');
  const fallbackSize = Number(values.get('--fallback-size') ?? 64);
  if (!Number.isSafeInteger(fallbackSize) || fallbackSize < 1 || fallbackSize > LIMITS.dimension) throw Error('TEXTURE_BUILD_FALLBACK_SIZE');
  const format = values.get('--format'), colorSpace = values.get('--color-space');
  const texconv = values.has('--texconv') ? resolve(values.get('--texconv')) : undefined;
  const encoder = values.get('--encoder') ?? (texconv ? 'texconv' : undefined);
  if (format !== undefined && !['rgba8', 'bc7'].includes(format)) throw Error('TEXTURE_BUILD_FORMAT');
  if (colorSpace !== undefined && !['srgb', 'linear'].includes(colorSpace)) throw Error('TEXTURE_BUILD_COLOR_SPACE');
  if (encoder !== undefined && !['texconv', 'builtin'].includes(encoder)) throw Error('TEXTURE_BUILD_ENCODER');
  if (extension === '.dds' && (texconv || encoder === 'texconv')) throw Error('TEXTURE_BUILD_DDS_DOES_NOT_RUN_CODEC');
  if (encoder === 'texconv' && (!texconv || extension !== '.png')) throw Error('TEXTURE_BUILD_TEXCONV_REQUIRES_PNG_AND_PATH');
  if (encoder === 'builtin' && texconv) throw Error('TEXTURE_BUILD_ENCODER_CONFLICT');
  if ((extension !== '.dds' || encoder) && (!format || !colorSpace || !encoder)) throw Error('TEXTURE_BUILD_IMAGE_REQUIRES_FORMAT_COLOR_SPACE_ENCODER');
  const gpuAdapter = values.has('--gpu-adapter') ? Number(values.get('--gpu-adapter')) : undefined;
  if (gpuAdapter !== undefined && (!Number.isSafeInteger(gpuAdapter) || gpuAdapter < 0 || gpuAdapter > 15
    || encoder !== 'texconv' || format !== 'bc7')) throw Error('TEXTURE_BUILD_GPU_ADAPTER');
  const builtinOnly = ['--workers', '--bc7-quality', '--bc7-early-out', '--bc7-channel-weights', '--normal-convention'].filter(key => values.has(key));
  if (builtinOnly.length && encoder !== 'builtin') throw Error('TEXTURE_BUILD_BUILTIN_OPTION');
  const workers = Number(values.get('--workers') ?? defaultWorkerCount());
  const bc7Quality = Number(values.get('--bc7-quality') ?? BUILTIN_DEFAULTS.bc7Quality);
  const bc7EarlyOutError = Number(values.get('--bc7-early-out') ?? BUILTIN_DEFAULTS.bc7EarlyOutError);
  if (!Number.isSafeInteger(workers) || workers < 0 || workers > 64) throw Error('TEXTURE_BUILD_WORKERS');
  if (![0, 1, 2].includes(bc7Quality)) throw Error('TEXTURE_BUILD_BC7_QUALITY');
  if (!Number.isSafeInteger(bc7EarlyOutError) || bc7EarlyOutError < 0 || bc7EarlyOutError > 1_000_000) throw Error('TEXTURE_BUILD_BC7_EARLY_OUT');
  const bc7ChannelWeights = (values.get('--bc7-channel-weights') ?? '1,1,1,1').split(',').map(Number);
  if (bc7ChannelWeights.length !== 4 || bc7ChannelWeights.some(w => !Number.isInteger(w) || w < 0 || w > 16) || bc7ChannelWeights.every(w => w === 0)) throw Error('TEXTURE_BUILD_BC7_CHANNEL_WEIGHTS');
  const normalConvention = values.get('--normal-convention');
  if (normalConvention !== undefined && (!['opengl', 'directx'].includes(normalConvention) || colorSpace !== 'linear')) throw Error('TEXTURE_BUILD_NORMAL_CONVENTION');
  return { input, output, id, fallbackSize, extension, format, colorSpace, texconv, gpuAdapter, encoder,
    workers, bc7Quality, bc7EarlyOutError, bc7ChannelWeights, normalConvention };
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) throw Error('TEXTURE_BUILD_OUTPUT_EXISTS');
  await mkdir(dirname(config.output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(config.output), '.' + basename(config.output) + '.tmp-'));
  const context = await createTextureContext();
  // workers 0 encodes on the main thread; output bytes are identical either way.
  const pool = config.encoder === 'builtin' && config.format === 'bc7' && config.workers > 0 ? new TextureCodecPool(config.workers) : null;
  try {
    const result = await buildTextureInto(config, temporary, context, pool);
    await publishDirectory(temporary, config.output);
    console.log(JSON.stringify(result));
  } finally {
    await pool?.close();
    await context.close();
    await removeOwnTemporary(temporary, dirname(config.output), config.output);
  }
}

main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
