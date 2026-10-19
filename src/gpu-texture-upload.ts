import type { TextureStorageLayout } from './texture-storage';

/** Uploads an existing texture's mip payloads in bounded, whole block-row writes.
 * The caller owns the texture and keeps the payloads alive until done. */
export class GpuTextureUpload {
  private mip = 0;
  private row = 0;
  private uploadedBytes = 0;
  private readonly blockSize: number;

  constructor(private readonly device: GPUDevice, private readonly texture: GPUTexture,
    private readonly layout: TextureStorageLayout, private readonly levels: readonly Uint8Array[]) {
    if (!layout.levels.length || layout.levels.length !== levels.length) {
      throw new Error('TEXTURE_UPLOAD_LEVEL_COUNT');
    }
    this.blockSize = layout.requiredFeature ? 4 : 1;
    let total = 0;
    for (let index = 0; index < layout.levels.length; index++) {
      const level = layout.levels[index], payload = levels[index];
      if (!(payload instanceof Uint8Array) || payload.byteLength !== level.byteLength ||
        !Number.isSafeInteger(level.bytesPerRow) || level.bytesPerRow < 1 ||
        !Number.isSafeInteger(level.rowsPerImage) || level.rowsPerImage < 1 ||
        level.byteLength !== level.bytesPerRow * level.rowsPerImage ||
        !Number.isSafeInteger(level.copyWidth) || level.copyWidth < 1 ||
        !Number.isSafeInteger(level.copyHeight) || level.copyHeight !== level.rowsPerImage * this.blockSize) {
        throw new Error('TEXTURE_UPLOAD_LEVEL_LAYOUT: mip ' + index);
      }
      total += level.byteLength;
    }
    if (!Number.isSafeInteger(total) || total !== layout.byteLength) {
      throw new Error('TEXTURE_UPLOAD_TOTAL_BYTES');
    }
  }

  get done(): boolean { return this.mip === this.layout.levels.length; }
  get remainingBytes(): number { return this.layout.byteLength - this.uploadedBytes; }
  get nextRowBytes(): number { return this.done ? 0 : this.layout.levels[this.mip].bytesPerRow; }

  /** Returns bytes submitted in this call; a budget below one row makes no progress. */
  upload(maxBytes: number): number {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new RangeError('TEXTURE_UPLOAD_BUDGET');
    let submitted = 0;
    while (!this.done) {
      const level = this.layout.levels[this.mip];
      const rows = Math.min(level.rowsPerImage - this.row,
        Math.floor((maxBytes - submitted) / level.bytesPerRow));
      if (rows === 0) break;
      const bytes = rows * level.bytesPerRow;
      const start = this.row * level.bytesPerRow;
      const payload = this.levels[this.mip].subarray(start, start + bytes);
      this.device.queue.writeTexture({ texture: this.texture, mipLevel: this.mip,
        origin: { x: 0, y: this.row * this.blockSize, z: 0 } }, payload,
      { bytesPerRow: level.bytesPerRow, rowsPerImage: rows },
      [level.copyWidth, rows * this.blockSize, 1]);
      // Advance only after writeTexture succeeds, so a thrown write can be retried.
      this.row += rows;
      this.uploadedBytes += bytes;
      submitted += bytes;
      if (this.row === level.rowsPerImage) { this.mip++; this.row = 0; }
    }
    return submitted;
  }
}
