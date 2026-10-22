/** Actual raster receiver inputs, shared by every geometry/deformation path.
 * The row is frame-local; temporal identity lives in the metadata tuple. */
export interface RasterReceiverBindings {
  readonly width: number;
  readonly height: number;
  readonly motionDepth: GPUTexture;
  readonly identityNormals: GPUTexture;
  readonly metadata: GPUBuffer;
  readonly metadataRows: number;
}

export const RASTER_RECEIVER_METADATA_BYTES = 32;
export const RASTER_RECEIVER_PIXEL_BYTES = 32;

export function rasterReceiverBytes(width: number, height: number, rows: number, budget: number,
  maxDimension = 8192, maxBufferBytes = 128 * 1024 * 1024): number {
  if (![width, height, rows, budget].every(Number.isSafeInteger) || width < 1 || height < 1 || rows < 1
    || width > maxDimension || height > maxDimension || rows > 0xffff_fffe
    || budget < 1 || budget > 512 * 1024 * 1024 || rows * RASTER_RECEIVER_METADATA_BYTES > maxBufferBytes) {
    throw new Error('INVALID_RASTER_RECEIVER_LAYOUT');
  }
  const bytes = width * height * RASTER_RECEIVER_PIXEL_BYTES + rows * RASTER_RECEIVER_METADATA_BYTES;
  if (!Number.isSafeInteger(bytes) || bytes > budget) throw new Error('RASTER_RECEIVER_BYTE_BUDGET');
  return bytes;
}
