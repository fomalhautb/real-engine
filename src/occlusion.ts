/** Bounded CPU configuration and allocation plan for a max-depth occlusion pyramid. */
export interface OcclusionOptions {
  readonly pixelSize?: number;
  readonly depthBias?: number;
  readonly maxBytes?: number;
}

export interface ResolvedOcclusionOptions {
  readonly pixelSize: 1 | 2 | 4 | 8;
  readonly depthBias: number;
  readonly maxBytes: number;
}

export interface OcclusionPyramidPlan {
  /** Padded base mip dimensions, after downsampling the viewport by pixelSize. */
  readonly width: number;
  readonly height: number;
  readonly pixelSize: 1 | 2 | 4 | 8;
  readonly mipLevelCount: number;
  /** Bytes in one r32float pyramid, including every mip down to 1x1. */
  readonly textureBytes: number;
  /** Two pyramids and 208 bytes of query/build parameters. */
  readonly bytes: number;
}

const MIB = 1024 * 1024;
const PARAMETER_BYTES = 2 * 96 + 16;

function boundedInteger(value: unknown, minimum: number, maximum: number, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`OCCLUSION_${name}`);
  }
  return value;
}

export function resolveOcclusionOptions(options: OcclusionOptions = {}): ResolvedOcclusionOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new RangeError('OCCLUSION_OPTIONS');
  const pixelSize = options.pixelSize === undefined ? 2 : options.pixelSize;
  if (pixelSize !== 1 && pixelSize !== 2 && pixelSize !== 4 && pixelSize !== 8) {
    throw new RangeError('OCCLUSION_PIXEL_SIZE');
  }
  const depthBias = options.depthBias === undefined ? 0.00002 : options.depthBias;
  if (typeof depthBias !== 'number' || !Number.isFinite(depthBias) || depthBias < 0 || depthBias > 0.01) {
    throw new RangeError('OCCLUSION_DEPTH_BIAS');
  }
  return Object.freeze({
    pixelSize,
    depthBias,
    maxBytes: boundedInteger(options.maxBytes === undefined ? 64 * MIB : options.maxBytes,
      1, 256 * MIB, 'MAX_BYTES'),
  });
}

function nextPowerOfTwo(value: number): number {
  let power = 1;
  while (power < value) power *= 2;
  return power;
}

/** Plan two complete r32float mip chains without silently increasing pixelSize. */
export function planOcclusionPyramid(
  viewportWidth: number, viewportHeight: number,
  options: OcclusionOptions | ResolvedOcclusionOptions = {},
): OcclusionPyramidPlan {
  boundedInteger(viewportWidth, 1, 16_384, 'WIDTH');
  boundedInteger(viewportHeight, 1, 16_384, 'HEIGHT');
  const { pixelSize, maxBytes } = resolveOcclusionOptions(options);
  const width = nextPowerOfTwo(Math.ceil(viewportWidth / pixelSize));
  const height = nextPowerOfTwo(Math.ceil(viewportHeight / pixelSize));
  let mipWidth = width, mipHeight = height, mipLevelCount = 0, texels = 0;
  for (;;) {
    texels += mipWidth * mipHeight;
    mipLevelCount++;
    if (mipWidth === 1 && mipHeight === 1) break;
    mipWidth = Math.max(1, mipWidth / 2);
    mipHeight = Math.max(1, mipHeight / 2);
  }
  const textureBytes = texels * 4;
  const bytes = 2 * textureBytes + PARAMETER_BYTES;
  if (bytes > maxBytes) throw new RangeError('OCCLUSION_BYTE_BUDGET');
  return Object.freeze({ width, height, pixelSize, mipLevelCount, textureBytes, bytes });
}
