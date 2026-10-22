import { rasterReceiverBytes, RASTER_RECEIVER_METADATA_BYTES, type RasterReceiverBindings } from './surface-receivers';

/** Two baseline-compatible 16-byte color attachments. The renderer supplies its
 * depth attachment, avoiding a third copy of visible depth. No CPU geometry readback. */
export class GpuSurfaceReceivers {
  readonly metadata: GPUBuffer;
  readonly data: Uint32Array<ArrayBuffer>;
  private motionDepth?: GPUTexture;
  private identityNormals?: GPUTexture;
  private width = 0;
  private height = 0;
  private bytes: number;
  private disposed = false;

  constructor(private device: GPUDevice, readonly metadataRows: number, readonly maxBytes: number,
    private account: (bytes: number) => void) {
    this.preflight(1, 1);
    if (device.limits.maxColorAttachmentBytesPerSample < 32 || device.limits.maxColorAttachments < 2) {
      throw new Error('RASTER_RECEIVER_ATTACHMENT_LIMIT');
    }
    this.bytes = metadataRows * RASTER_RECEIVER_METADATA_BYTES;
    this.data = new Uint32Array(metadataRows * 8);
    this.metadata = device.createBuffer({ label: 'raster receiver stable identity metadata', size: this.bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    account(this.bytes);
  }

  preflight(width: number, height: number): number {
    const bytes = rasterReceiverBytes(width, height, this.metadataRows, this.maxBytes, this.device.limits.maxTextureDimension2D,
      Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize));
    // A resize prepares a complete replacement before publishing it. Its
    // overlap with the previous attachments is part of the same hard budget.
    const oldPixels = width === this.width && height === this.height ? 0 : this.width * this.height * 32;
    if (bytes + oldPixels > this.maxBytes) throw new Error('RASTER_RECEIVER_RESIZE_BYTE_BUDGET');
    return bytes;
  }

  resize(width: number, height: number): RasterReceiverBindings {
    if (this.disposed) throw new Error('RASTER_RECEIVER_DISPOSED');
    if (width !== this.width || height !== this.height) {
      const bytes = this.preflight(width, height);
      const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
      const prepared: GPUTexture[] = [];
      try {
        for (const format of ['rgba32float', 'rgba32uint'] as const) {
          prepared.push(this.device.createTexture({ label: 'raster receiver ' + format,
            size: [width, height], format, usage }));
          this.account(width * height * 16);
        }
      } catch (error) {
        for (const texture of prepared) { texture.destroy(); this.account(-width * height * 16); }
        throw error;
      }
      this.motionDepth?.destroy(); this.identityNormals?.destroy();
      this.account(-this.width * this.height * 32);
      [this.motionDepth, this.identityNormals] = prepared;
      this.bytes = bytes;
      this.width = width; this.height = height;
    }
    return { width, height, motionDepth: this.motionDepth!, identityNormals: this.identityNormals!,
      metadata: this.metadata, metadataRows: this.metadataRows };
  }

  upload(): number {
    if (this.disposed) throw new Error('RASTER_RECEIVER_DISPOSED');
    this.device.queue.writeBuffer(this.metadata, 0, this.data);
    return this.data.byteLength;
  }

  attachments(recovery: boolean): GPURenderPassColorAttachment[] {
    return [this.motionDepth!, this.identityNormals!].map(texture => ({ view: texture.createView(),
      clearValue: [0, 0, 0, 0], loadOp: recovery ? 'load' : 'clear', storeOp: 'store' }));
  }

  diagnostics() {
    return { width: this.width, height: this.height, metadataRows: this.metadataRows, ownedGpuBytes: this.bytes,
      maxBytes: this.maxBytes, pixelBytes: 32, motionDepthFormat: 'rgba32float', identityNormalsFormat: 'rgba32uint',
      geometry: 'current-raster-selection', normals: 'shared-material-gradients', representation: 'stable-cluster-id' };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.motionDepth?.destroy(); this.identityNormals?.destroy(); this.metadata.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
