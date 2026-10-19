import type { PreparedTextureFormat } from './texture-storage';
import { createHttpTextureMipProvider, validateTextureStreamSource, verifyTextureStreamMip } from './texture-stream-source';
import type { TextureStreamMip, TextureStreamSource } from './texture-stream-types';

export interface TextureStreamManifest {
  readonly version: 1;
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly colorSpace: 'srgb' | 'linear';
  readonly format: PreparedTextureFormat;
  readonly fallbackMip: number;
  readonly mips: readonly (TextureStreamMip & { readonly file: string })[];
}

export interface TextureStreamLoadOptions {
  readonly signal?: AbortSignal;
  readonly maxManifestBytes?: number;
  readonly maxFallbackBytes?: number;
}

const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_FALLBACK_BYTES = 64 * 1024 * 1024;
const DEFAULT_MANIFEST_BYTES = 64 * 1024;
const DEFAULT_FALLBACK_BYTES = 4 * 1024 * 1024;
const FORMATS: readonly string[] = ['rgba8', 'bc7', 'etc2-rgba8', 'astc-4x4'];

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function dimension(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 1 && (value as number) <= 8192;
}

function mipSize(base: number, level: number): number {
  return Math.max(1, Math.floor(base / 2 ** level));
}

function safeFile(file: unknown): file is string {
  return typeof file === 'string' && file.length > 0 && file.length <= 1024
    && file.split('/').every(segment => segment !== '.' && segment !== '..'
      && /^[A-Za-z0-9._-]+$/.test(segment));
}

function abortIfNeeded(signal: AbortSignal): void {
  if (signal.aborted) throw signal.reason ?? new DOMException('Texture stream load aborted', 'AbortError');
}

function budget(value: number | undefined, fallback: number, ceiling: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > ceiling) {
    throw new RangeError(`TEXTURE_STREAM: ${label} must be 1-${ceiling} bytes`);
  }
  return result;
}

/** Owns and freezes all manifest metadata; no input objects are retained. */
export function validateTextureStreamManifest(input: unknown): TextureStreamManifest {
  if (!record(input) || input.version !== 1 || typeof input.id !== 'string'
    || input.id.trim().length < 1 || input.id.length > 128 || /[\x00-\x1f\x7f]/.test(input.id)
    || !dimension(input.width) || !dimension(input.height)
    || (input.colorSpace !== 'srgb' && input.colorSpace !== 'linear')
    || !FORMATS.includes(input.format as string)) {
    throw new TypeError('TEXTURE_STREAM: invalid manifest metadata');
  }
  const width = input.width;
  const height = input.height;
  const format = input.format as PreparedTextureFormat;
  const count = 1 + Math.floor(Math.log2(Math.max(width, height)));
  if (!Array.isArray(input.mips) || input.mips.length !== count
    || !Number.isSafeInteger(input.fallbackMip) || (input.fallbackMip as number) < 0
    || (input.fallbackMip as number) >= count) {
    throw new RangeError('TEXTURE_STREAM: mips must form a complete chain through 1x1');
  }
  const fallbackMip = input.fallbackMip as number;
  if (format !== 'rgba8' && (width % 4 !== 0 || height % 4 !== 0
    || mipSize(width, fallbackMip) % 4 !== 0 || mipSize(height, fallbackMip) % 4 !== 0)) {
    throw new RangeError('TEXTURE_STREAM: compressed base and fallback dimensions must be divisible by four');
  }
  const mips = input.mips.map((raw: unknown, level: number) => {
    const w = mipSize(width, level);
    const h = mipSize(height, level);
    const expected = format === 'rgba8' ? w * h * 4 : Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    if (!record(raw) || raw.level !== level || raw.width !== w || raw.height !== h
      || raw.byteLength !== expected || typeof raw.sha256 !== 'string'
      || !/^[\da-f]{64}$/i.test(raw.sha256) || !safeFile(raw.file)) {
      throw new RangeError(`TEXTURE_STREAM: invalid manifest mip ${level}`);
    }
    return Object.freeze({ level, width: w, height: h, byteLength: expected,
      sha256: raw.sha256.toLowerCase(), file: raw.file });
  });
  return Object.freeze({ version: 1, id: input.id, width, height,
    colorSpace: input.colorSpace, format, fallbackMip, mips: Object.freeze(mips) });
}

async function readManifest(response: Response, signal: AbortSignal, limit: number): Promise<Uint8Array> {
  if (!response.ok || !response.body) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error(`TEXTURE_STREAM: HTTP ${response.status} or empty manifest body`);
  }
  const declared = response.headers?.get('content-length');
  if (declared !== null && declared !== undefined && /^\d+$/.test(declared) && Number(declared) > limit) {
    await response.body.cancel().catch(() => undefined);
    throw new RangeError('TEXTURE_STREAM: oversized manifest');
  }
  const reader = response.body.getReader();
  const bytes = new Uint8Array(limit);
  let length = 0;
  const cancelOnAbort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
  signal.addEventListener('abort', cancelOnAbort, { once: true });
  try {
    while (true) {
      abortIfNeeded(signal);
      const chunk = await reader.read();
      abortIfNeeded(signal);
      if (chunk.done) break;
      if (!(chunk.value instanceof Uint8Array) || chunk.value.byteLength > limit - length) {
        throw new RangeError('TEXTURE_STREAM: oversized manifest');
      }
      bytes.set(chunk.value, length);
      length += chunk.value.byteLength;
    }
    return bytes.slice(0, length);
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    signal.removeEventListener('abort', cancelOnAbort);
    reader.releaseLock();
  }
}

/** Loads and verifies the complete resident tail before exposing a streaming source. */
export async function loadTextureStreamSource(
  manifestUrl: string, options: TextureStreamLoadOptions = {},
): Promise<TextureStreamSource> {
  const signal = options.signal ?? new AbortController().signal;
  const manifestLimit = budget(options.maxManifestBytes, DEFAULT_MANIFEST_BYTES, MAX_MANIFEST_BYTES, 'maxManifestBytes');
  const fallbackLimit = budget(options.maxFallbackBytes, DEFAULT_FALLBACK_BYTES, MAX_FALLBACK_BYTES, 'maxFallbackBytes');
  abortIfNeeded(signal);
  const requestUrl = new URL(manifestUrl, globalThis.location?.href);
  if (!/^https?:$/.test(requestUrl.protocol) || requestUrl.username || requestUrl.password) {
    throw new TypeError('TEXTURE_STREAM: manifest URL must be HTTP(S)');
  }
  const response = await fetch(requestUrl.href, { signal });
  if (signal.aborted) {
    await response.body?.cancel().catch(() => undefined);
    abortIfNeeded(signal);
  }
  const finalUrl = new URL(response.url || requestUrl.href);
  if (finalUrl.origin !== requestUrl.origin || !/^https?:$/.test(finalUrl.protocol)) {
    await response.body?.cancel().catch(() => undefined);
    throw new TypeError('TEXTURE_STREAM: manifest redirect changed origin');
  }
  const bytes = await readManifest(response, signal, manifestLimit);
  abortIfNeeded(signal);
  const manifest = validateTextureStreamManifest(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)));
  let fallbackBytes = 0;
  for (let level = manifest.fallbackMip; level < manifest.mips.length; level++) {
    fallbackBytes += manifest.mips[level].byteLength;
    if (fallbackBytes > fallbackLimit) throw new RangeError('TEXTURE_STREAM: fallback byte budget exceeded');
  }
  const urls = manifest.mips.map(mip => {
    const url = new URL(mip.file, finalUrl);
    if (url.origin !== finalUrl.origin) throw new TypeError('TEXTURE_STREAM: mip changed manifest origin');
    return url.href;
  });
  const provider = createHttpTextureMipProvider(urls);
  const levels: { width: number; height: number; data: Uint8Array }[] = [];
  for (let level = manifest.fallbackMip; level < manifest.mips.length; level++) {
    abortIfNeeded(signal);
    const mip = manifest.mips[level];
    const data = await verifyTextureStreamMip(mip, await provider(mip, signal), signal);
    levels.push({ width: mip.width, height: mip.height, data });
  }
  abortIfNeeded(signal);
  return validateTextureStreamSource({ id: manifest.id, width: manifest.width, height: manifest.height,
    colorSpace: manifest.colorSpace, format: manifest.format, mips: manifest.mips,
    fallback: { kind: 'prepared', width: manifest.width, height: manifest.height,
      colorSpace: manifest.colorSpace, format: manifest.format, baseMipLevel: manifest.fallbackMip, levels },
    provider });
}
