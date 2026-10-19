import { textureStorageLayout } from './texture-storage';
import type { TextureMipProvider, TextureStreamMip, TextureStreamSource } from './texture-stream-types';

const MAX_DIMENSION = 8192;
const MAX_LEVELS = 14;
const MAX_ID_LENGTH = 128;
const MAX_MIP_BYTES = MAX_DIMENSION * MAX_DIMENSION * 4;

function abortIfNeeded(signal: AbortSignal): void {
  if (signal.aborted) throw signal.reason ?? new DOMException('Texture mip request aborted', 'AbortError');
}

function expectedMipBytes(width: number, height: number, format: TextureStreamSource['format']): number {
  if (format === 'rgba8') return width * height * 4;
  if (format === 'bc7' || format === 'etc2-rgba8' || format === 'astc-4x4') {
    return Math.ceil(width / 4) * Math.ceil(height / 4) * 16;
  }
  throw new TypeError('TEXTURE_STREAM: unsupported format');
}

function validDimensions(width: number, height: number): boolean {
  return Number.isSafeInteger(width) && Number.isSafeInteger(height)
    && width >= 1 && height >= 1 && width <= MAX_DIMENSION && height <= MAX_DIMENSION;
}

function mipDimension(dimension: number, level: number): number {
  return Math.max(1, Math.floor(dimension / 2 ** level));
}

function validNativeMipSize(mip: TextureStreamMip): boolean {
  return Number.isSafeInteger(mip.level) && mip.level >= 0 && mip.level < MAX_LEVELS
    && validDimensions(mip.width, mip.height)
    && Number.isSafeInteger(mip.byteLength) && mip.byteLength > 0 && mip.byteLength <= MAX_MIP_BYTES
    && (mip.byteLength === expectedMipBytes(mip.width, mip.height, 'rgba8')
      || mip.byteLength === expectedMipBytes(mip.width, mip.height, 'bc7'));
}

/** Validates the complete native mip manifest without copying or allocating image payloads. */
export function validateTextureStreamSource(source: TextureStreamSource): TextureStreamSource {
  if (!source || typeof source !== 'object') throw new TypeError('TEXTURE_STREAM: expected source');
  if (typeof source.id !== 'string' || source.id.trim().length < 1 || source.id.length > MAX_ID_LENGTH
    || /[\x00-\x1f\x7f]/.test(source.id)) {
    throw new TypeError(`TEXTURE_STREAM: id must be 1-${MAX_ID_LENGTH} printable characters`);
  }
  if (!validDimensions(source.width, source.height)) {
    throw new RangeError(`TEXTURE_STREAM: dimensions must be integers from 1 to ${MAX_DIMENSION}`);
  }
  if (source.colorSpace !== 'srgb' && source.colorSpace !== 'linear') {
    throw new TypeError('TEXTURE_STREAM: unsupported colorSpace');
  }
  if (typeof source.provider !== 'function') throw new TypeError('TEXTURE_STREAM: provider must be a function');
  const count = 1 + Math.floor(Math.log2(Math.max(source.width, source.height)));
  if (count > MAX_LEVELS || !Array.isArray(source.mips) || source.mips.length !== count) {
    throw new RangeError('TEXTURE_STREAM: mips must form a complete chain through 1x1');
  }
  if (source.format !== 'rgba8' && (source.width % 4 !== 0 || source.height % 4 !== 0)) {
    throw new RangeError('TEXTURE_STREAM: compressed base dimensions must be divisible by four');
  }
  const mips = source.mips.map((mip, level) => {
    const width = mipDimension(source.width, level);
    const height = mipDimension(source.height, level);
    if (!mip || mip.level !== level || mip.width !== width || mip.height !== height
      || mip.byteLength !== expectedMipBytes(width, height, source.format)
      || typeof mip.sha256 !== 'string' || !/^[\da-f]{64}$/i.test(mip.sha256)) {
      throw new RangeError(`TEXTURE_STREAM: invalid native mip ${level}`);
    }
    return Object.freeze({ level, width, height, byteLength: mip.byteLength, sha256: mip.sha256.toLowerCase() });
  });
  const fallback = source.fallback;
  const layout = textureStorageLayout(fallback);
  if (fallback.kind !== 'prepared' || fallback.width !== source.width || fallback.height !== source.height
    || fallback.colorSpace !== source.colorSpace || fallback.format !== source.format
    || layout.baseMipLevel >= count) {
    throw new TypeError('TEXTURE_STREAM: fallback must match source storage');
  }
  for (let index = 0; index < fallback.levels.length; index++) {
    const mip = mips[fallback.baseMipLevel + index];
    const level = fallback.levels[index];
    if (!mip || level.width !== mip.width || level.height !== mip.height || level.data.byteLength !== mip.byteLength) {
      throw new RangeError(`TEXTURE_STREAM: fallback mip ${fallback.baseMipLevel + index} mismatches manifest`);
    }
  }
  return Object.freeze({ id: source.id, width: source.width, height: source.height,
    colorSpace: source.colorSpace, format: source.format, mips: Object.freeze(mips),
    fallback, provider: source.provider });
}

/** Hashes only an owned, exact-size native payload so accounting matches retained memory. */
export async function verifyTextureStreamMip(
  mip: TextureStreamMip, data: Uint8Array, signal: AbortSignal,
): Promise<Uint8Array> {
  abortIfNeeded(signal);
  if (!mip || !validNativeMipSize(mip)
    || typeof mip.sha256 !== 'string' || !/^[\da-f]{64}$/i.test(mip.sha256)) {
    throw new TypeError('TEXTURE_STREAM: invalid mip digest metadata');
  }
  if (!(data instanceof Uint8Array) || data.byteLength !== mip.byteLength
    || !(data.buffer instanceof ArrayBuffer) || data.byteOffset !== 0
    || data.buffer.byteLength !== mip.byteLength) {
    throw new TypeError(`TEXTURE_STREAM: mip ${mip.level} requires an owned ${mip.byteLength}-byte buffer`);
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data.buffer));
  abortIfNeeded(signal);
  const hash = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
  if (hash !== mip.sha256.toLowerCase()) throw new Error(`TEXTURE_STREAM: SHA-256 mismatch for mip ${mip.level}`);
  return data;
}

/** Checks the supplied resident tail against the same manifest used for fetched mips. */
export async function verifyTextureStreamFallback(source: TextureStreamSource, signal: AbortSignal): Promise<void> {
  const validated = validateTextureStreamSource(source);
  for (let index = 0; index < validated.fallback.levels.length; index++) {
    abortIfNeeded(signal);
    await verifyTextureStreamMip(validated.mips[validated.fallback.baseMipLevel + index],
      validated.fallback.levels[index].data, signal);
  }
}

/** Fetches one native mip at a time with a strict byte limit and cancellable body. */
export function createHttpTextureMipProvider(urls: readonly string[]): TextureMipProvider {
  if (!Array.isArray(urls) || urls.length < 1 || urls.length > MAX_LEVELS
    || urls.some(url => typeof url !== 'string' || url.length === 0)) {
    throw new TypeError('TEXTURE_STREAM: expected 1-14 mip URLs');
  }
  const mipUrls = Object.freeze([...urls]);
  return async (mip, signal) => {
    abortIfNeeded(signal);
    if (!mip || !validNativeMipSize(mip) || mip.level >= mipUrls.length) {
      throw new RangeError('TEXTURE_STREAM: invalid HTTP mip request');
    }
    const response = await fetch(mipUrls[mip.level], { signal });
    if (signal.aborted) {
      await response.body?.cancel().catch(() => undefined);
      abortIfNeeded(signal);
    }
    if (!response.ok || !response.body) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`TEXTURE_STREAM: HTTP ${response.status} or empty body for mip ${mip.level}`);
    }
    const reader = response.body.getReader();
    const result = new Uint8Array(mip.byteLength);
    let length = 0;
    const cancelOnAbort = () => { void reader.cancel(signal.reason).catch(() => undefined); };
    signal.addEventListener('abort', cancelOnAbort, { once: true });
    try {
      while (true) {
        abortIfNeeded(signal);
        const chunk = await reader.read();
        abortIfNeeded(signal);
        if (chunk.done) break;
        if (!(chunk.value instanceof Uint8Array) || chunk.value.byteLength > result.byteLength - length) {
          throw new RangeError(`TEXTURE_STREAM: oversized HTTP mip ${mip.level}`);
        }
        result.set(chunk.value, length);
        length += chunk.value.byteLength;
      }
      if (length !== result.byteLength) throw new RangeError(`TEXTURE_STREAM: short HTTP mip ${mip.level}`);
      return result;
    } catch (error) {
      await reader.cancel().catch(() => undefined);
      throw error;
    } finally {
      signal.removeEventListener('abort', cancelOnAbort);
      reader.releaseLock();
    }
  };
}
