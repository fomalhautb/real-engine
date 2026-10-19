import type { PreparedTextureAsset, PreparedTextureFormat } from './texture-storage';

/** One independently addressable native mip, identified by its exact bytes. */
export interface TextureStreamMip {
  readonly level: number;
  readonly width: number;
  readonly height: number;
  readonly byteLength: number;
  readonly sha256: string;
}
export type TextureMipProvider = (mip: TextureStreamMip, signal: AbortSignal) => Promise<Uint8Array>;

/** Materials use fallback as stable source identity. The renderer resolves it to
 * current residency, including later asynchronous material changes. Source and
 * payloads are immutable; every mip, including the supplied fallback, is hashed. */
export interface TextureStreamSource {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly colorSpace: 'srgb' | 'linear';
  readonly format: PreparedTextureFormat;
  readonly mips: readonly TextureStreamMip[];
  readonly fallback: PreparedTextureAsset;
  readonly provider: TextureMipProvider;
}

export interface TextureStreamingOptions {
  readonly sources: readonly TextureStreamSource[];
  /** Includes root payloads, retained current detail and reserved incoming mips.
   * Provider/hash working memory is additionally reserved in maxInFlightBytes. */
  readonly maxCpuBytes?: number;
  readonly maxInFlightBytes?: number;
  /** Global native texture bytes submitted by streaming per rendered frame. */
  readonly maxUploadBytesPerFrame?: number;
  /** Consecutive frames of lower demand before releasing detail. */
  readonly downgradeDelayFrames?: number;
}

export interface TextureStreamDemand {
  readonly source: PreparedTextureAsset;
  readonly mip: number;
  readonly priority: number;
}
