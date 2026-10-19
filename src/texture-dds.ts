import { MAX_TEXTURE_DIMENSION } from './texture-assets';
import type { PreparedTextureAsset } from './texture-storage';

const HEADER_BYTES = 148; // Magic + DDS_HEADER + DDS_HEADER_DXT10.
const DDS_MAGIC = 0x20534444;
const DX10_FOURCC = 0x30315844;
const DDSD_PITCH = 0x8;
const DDSD_LINEARSIZE = 0x80000;
const DDSD_DEPTH = 0x800000;
const DDSCAPS2_CUBEMAP_FACES = 0xfe00;
const DDSCAPS2_VOLUME = 0x200000;
const DDS_RESOURCE_MISC_TEXTURECUBE = 0x4;

function invalid(reason: string): never {
  throw new RangeError(`TEXTURE_DDS: ${reason}`);
}

/** Reads one tightly packed DX10 2D texture. No decoding or mip generation occurs. */
export function parseTextureDds(bytes: Uint8Array): PreparedTextureAsset {
  if (!(bytes instanceof Uint8Array)) throw new TypeError('TEXTURE_DDS: expected Uint8Array');
  if (bytes.byteLength < HEADER_BYTES) invalid('truncated DX10 header');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (offset: number): number => view.getUint32(offset, true);

  if (u32(0) !== DDS_MAGIC || u32(4) !== 124 || u32(76) !== 32) invalid('invalid DDS header');
  const flags = u32(8);
  const height = u32(12);
  const width = u32(16);
  const pitchOrLinearSize = u32(20);
  const depth = u32(24);
  const mipCount = u32(28);
  const pixelFlags = u32(80);
  if ((pixelFlags & 0x4) === 0 || u32(84) !== DX10_FOURCC) invalid('DX10 pixel format required');
  if (u32(112) & (DDSCAPS2_CUBEMAP_FACES | DDSCAPS2_VOLUME)) invalid('cube and volume textures are unsupported');

  const dxgiFormat = u32(128);
  let format: PreparedTextureAsset['format'];
  let colorSpace: PreparedTextureAsset['colorSpace'];
  switch (dxgiFormat) {
    case 28: format = 'rgba8'; colorSpace = 'linear'; break;
    case 29: format = 'rgba8'; colorSpace = 'srgb'; break;
    case 98: format = 'bc7'; colorSpace = 'linear'; break;
    case 99: format = 'bc7'; colorSpace = 'srgb'; break;
    default: return invalid(`unsupported DXGI format ${dxgiFormat}`);
  }
  if (u32(132) !== 3 || u32(140) !== 1 || (u32(136) & DDS_RESOURCE_MISC_TEXTURECUBE) !== 0
    || (flags & DDSD_DEPTH) !== 0 || depth > 1) invalid('only a single 2D texture is supported');
  const alphaMode = u32(144) & 0x7;
  if (alphaMode === 2 || alphaMode > 4) invalid('unsupported alpha mode');
  if (width < 1 || height < 1 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION) {
    invalid(`dimensions must be 1..${MAX_TEXTURE_DIMENSION}`);
  }
  if (format === 'bc7' && (width % 4 !== 0 || height % 4 !== 0)) {
    invalid('BC7 base dimensions must be divisible by four');
  }
  const expectedMips = 1 + Math.floor(Math.log2(Math.max(width, height)));
  if (mipCount !== expectedMips) invalid('incomplete or invalid mip chain');

  const blockCompressed = format === 'bc7';
  const topRowBytes = blockCompressed ? Math.ceil(width / 4) * 16 : width * 4;
  const topLevelBytes = topRowBytes * (blockCompressed ? Math.ceil(height / 4) : height);
  const hasPitch = (flags & DDSD_PITCH) !== 0;
  const hasLinearSize = (flags & DDSD_LINEARSIZE) !== 0;
  if (hasPitch && hasLinearSize) invalid('ambiguous pitch and linear size flags');
  if (hasPitch && (blockCompressed || pitchOrLinearSize !== topRowBytes)) invalid('unsupported row pitch');
  if (hasLinearSize && (!blockCompressed || pitchOrLinearSize !== topLevelBytes)) invalid('unsupported linear size');
  if (!hasPitch && !hasLinearSize && pitchOrLinearSize !== 0) invalid('unmarked pitch or linear size');

  const layouts: { width: number; height: number; byteLength: number }[] = [];
  let payloadLength = 0;
  let mipWidth = width;
  let mipHeight = height;
  for (let mip = 0; mip < mipCount; mip++) {
    const byteLength = blockCompressed
      ? Math.ceil(mipWidth / 4) * Math.ceil(mipHeight / 4) * 16
      : mipWidth * mipHeight * 4;
    layouts.push({ width: mipWidth, height: mipHeight, byteLength });
    payloadLength += byteLength;
    mipWidth = Math.max(1, Math.floor(mipWidth / 2));
    mipHeight = Math.max(1, Math.floor(mipHeight / 2));
  }
  if (bytes.byteLength !== HEADER_BYTES + payloadLength) invalid('truncated or extra mip payload bytes');

  let offset = HEADER_BYTES;
  const levels = layouts.map(({ width: levelWidth, height: levelHeight, byteLength }) => {
    const data = new Uint8Array(bytes.subarray(offset, offset + byteLength));
    offset += byteLength;
    return { width: levelWidth, height: levelHeight, data };
  });
  return { kind: 'prepared', width, height, baseMipLevel: 0, format, colorSpace, levels };
}
