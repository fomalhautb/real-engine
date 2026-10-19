import type { MaterialBinding } from './contracts';
import type { MaterialTexturePin, PreparedTextureUpload, TextureReplacementResult } from './gpu-material';
import type { MaterialTextureAsset, PreparedTextureAsset } from './texture-storage';
import { textureStorageLayout } from './texture-storage';
import { validateTextureStreamSource, verifyTextureStreamFallback, verifyTextureStreamMip } from './texture-stream-source';
import type { TextureStreamDemand, TextureStreamSource, TextureStreamingOptions } from './texture-stream-types';

export interface TextureStreamingBackend {
  readonly budgetBytes: number;
  imageBytes(): number;
  pin(asset: PreparedTextureAsset): MaterialTexturePin;
  stage(asset: PreparedTextureAsset): PreparedTextureUpload;
  replace(source: PreparedTextureAsset, next: PreparedTextureAsset): TextureReplacementResult;
}
type Entry = {
  source: TextureStreamSource; current: PreparedTextureAsset; pin: MaterialTexturePin;
  bytes: number[]; allowed: number[]; wanted: number; planned: number; priority: number;
  lowerDemandFrames: number; failure?: { mip: number; message: string };
};
type Transfer = {
  entry: Entry; mip: number; bridge: boolean; controller: AbortController;
  reserved: number; loading: boolean; cancelled: boolean; promise?: Promise<void>;
  asset?: PreparedTextureAsset; upload?: PreparedTextureUpload;
};
function option(value: number | undefined, fallback: number, maximum: number, label: string, minimum = 1): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < minimum || result > maximum) throw new Error('TEXTURE_STREAM_OPTION: ' + label);
  return result;
}

/** One globally bounded transfer at a time. A complete low-mip image stays bound
 * throughout IO/upload; all currently retained material references publish at a
 * frame boundary. Demand is independent of material evaluation and world queries. */
export class TextureStreaming {
  private readonly entries: Entry[] = [];
  private readonly identities = new Map<MaterialTextureAsset, Entry>();
  private readonly maxCpuBytes: number;
  private readonly maxInFlightBytes: number;
  private readonly maxUploadBytes: number;
  private readonly delay: number;
  private rootBytes = 0;
  private scratchBytes = 0;
  private transfer?: Transfer;
  private disposed = false;
  private disposal?: Promise<void>;
  private inFlightBytes = 0;
  private requests = 0;
  private cancellations = 0;
  private publications = 0;
  private evictions = 0;
  private uploadedBytes = 0;
  private frameUploadedBytes = 0;
  private budgetBlocked = false;

  static async create(options: TextureStreamingOptions, backend: TextureStreamingBackend): Promise<TextureStreaming> {
    const result = new TextureStreaming(options, backend);
    try {
      const signal = new AbortController().signal;
      for (const entry of result.entries) {
        await verifyTextureStreamFallback(entry.source, signal);
        entry.pin = backend.pin(entry.current);
      }
      return result;
    } catch (error) { await result.dispose(); throw error; }
  }

  private constructor(options: TextureStreamingOptions, private readonly backend: TextureStreamingBackend) {
    this.maxCpuBytes = option(options.maxCpuBytes, 128 * 1024 * 1024, 2 ** 31, 'maxCpuBytes');
    this.maxInFlightBytes = option(options.maxInFlightBytes, 64 * 1024 * 1024, 2 ** 31, 'maxInFlightBytes');
    this.maxUploadBytes = option(options.maxUploadBytesPerFrame, 256 * 1024, 64 * 1024 * 1024, 'maxUploadBytesPerFrame');
    this.delay = option(options.downgradeDelayFrames, 15, 3600, 'downgradeDelayFrames', 0);
    if (!Array.isArray(options.sources) || options.sources.length < 1 || options.sources.length > 1024) throw new Error('TEXTURE_STREAM_SOURCE_LIMIT');
    const ids = new Set<string>();
    for (const input of options.sources) {
      const source = validateTextureStreamSource(input);
      if (ids.has(source.id) || this.identities.has(source.fallback)) throw new Error('TEXTURE_STREAM_DUPLICATE_SOURCE');
      ids.add(source.id);
      const bytes = source.mips.map((_, index) => source.mips.slice(index).reduce((sum, mip) => sum + mip.byteLength, 0));
      const allowed = source.mips.filter(mip => mip.level <= source.fallback.baseMipLevel &&
        (source.format === 'rgba8' || mip.width % 4 === 0 && mip.height % 4 === 0)).map(mip => mip.level);
      for (const mip of source.mips) {
        const row = source.format === 'rgba8' ? mip.width * 4 : Math.ceil(mip.width / 4) * 16;
        if (row > this.maxUploadBytes || 2 * mip.byteLength > this.maxInFlightBytes) throw new Error('TEXTURE_STREAM_MIP_BUDGET: ' + source.id);
      }
      // Root re-uploads also obey the same per-frame row budget.
      if (textureStorageLayout(source.fallback).levels.some(level => level.bytesPerRow > this.maxUploadBytes)) throw new Error('TEXTURE_STREAM_ROW_BUDGET');
      const root = bytes[source.fallback.baseMipLevel];
      this.rootBytes += root; this.scratchBytes = Math.max(this.scratchBytes, root);
      const entry: Entry = { source, current: source.fallback, pin: undefined as unknown as MaterialTexturePin,
        bytes, allowed, wanted: source.fallback.baseMipLevel, planned: source.fallback.baseMipLevel,
        priority: 0, lowerDemandFrames: 0 };
      this.entries.push(entry); this.identities.set(source.fallback, entry);
    }
    if (this.rootBytes > this.maxCpuBytes || this.rootBytes + this.scratchBytes + backend.imageBytes() > backend.budgetBytes) {
      throw new Error('TEXTURE_STREAM_ROOT_BUDGET');
    }
  }

  /** Preserve canonical fallback identity in game bindings; resolve only when a
   * GPU material is created. A pending material is included in atomic publication. */
  resolve(binding: MaterialBinding | undefined): MaterialBinding | undefined {
    this.live();
    if (!binding?.textures) return binding;
    return { ...binding, textures: binding.textures.map(asset => asset ? this.identities.get(asset)?.current ?? asset : asset) };
  }
  source(asset: MaterialTextureAsset | undefined): TextureStreamSource | undefined {
    return asset ? this.identities.get(asset)?.source : undefined;
  }
  owns(asset: MaterialTextureAsset): boolean {
    return this.identities.has(asset) || this.entries.some(entry => entry.current === asset) || this.transfer?.asset === asset;
  }
  private live(): void { if (this.disposed) throw new Error('TEXTURE_STREAM_DISPOSED'); }
  private cpuBytes(): number {
    return this.rootBytes + this.entries.reduce((sum, entry) => sum + entry.bytes[entry.current.baseMipLevel] - entry.bytes[entry.source.fallback.baseMipLevel], 0);
  }
  private clamp(entry: Entry, mip: number): number {
    // A compressed base cannot start at an unaligned tiny mip. Prefer finer data.
    let result = entry.allowed[0];
    for (const allowed of entry.allowed) if (allowed <= mip) result = allowed;
    return result;
  }

  private plan(demands: readonly TextureStreamDemand[]): void {
    if (!Array.isArray(demands) || demands.length > 4096) throw new Error('TEXTURE_STREAM_DEMAND_LIMIT');
    const desired = new Map<Entry, { mip: number; priority: number }>();
    for (const demand of demands) {
      const entry = this.identities.get(demand.source);
      if (!entry || !Number.isSafeInteger(demand.mip) || demand.mip < 0 || demand.mip >= entry.source.mips.length ||
        !Number.isFinite(demand.priority) || demand.priority < 0) throw new Error('TEXTURE_STREAM_DEMAND');
      const old = desired.get(entry), mip = this.clamp(entry, demand.mip);
      desired.set(entry, { mip: Math.min(mip, old?.mip ?? mip), priority: Math.max(demand.priority, old?.priority ?? 0) });
    }
    for (const entry of this.entries) {
      const demand = desired.get(entry), mip = demand?.mip ?? entry.source.fallback.baseMipLevel;
      entry.priority = demand?.priority ?? 0;
      if (mip > entry.wanted) {
        entry.lowerDemandFrames++;
        if (entry.lowerDemandFrames >= this.delay) { entry.wanted = mip; entry.lowerDemandFrames = 0; }
      } else { entry.wanted = mip; entry.lowerDemandFrames = 0; }
      entry.planned = entry.source.fallback.baseMipLevel;
    }
    const currentBytes = this.entries.reduce((sum, entry) => sum + entry.bytes[entry.current.baseMipLevel], 0);
    const stagedBytes = this.transfer?.upload?.diagnostics().allocationBytes ?? 0;
    const unmanaged = Math.max(0, this.backend.imageBytes() - currentBytes - stagedBytes);
    let available = Math.min(this.backend.budgetBytes - unmanaged - this.scratchBytes, this.maxCpuBytes) - this.rootBytes;
    this.budgetBlocked = available < 0;
    available = Math.max(0, available);
    for (const entry of [...this.entries].sort((a, b) => b.priority - a.priority || a.source.id.localeCompare(b.source.id))) {
      const root = entry.bytes[entry.source.fallback.baseMipLevel];
      for (const mip of entry.allowed) {
        if (mip < entry.wanted || entry.bytes[mip] - root > available) continue;
        entry.planned = mip; available -= entry.bytes[mip] - root; break;
      }
    }
  }

  /** Called once per rendered frame, before any pass consumes material bindings. */
  tick(demands: readonly TextureStreamDemand[]): number {
    this.live(); this.frameUploadedBytes = 0; this.plan(demands);
    const previous = this.transfer;
    if (previous && !previous.cancelled && !previous.bridge && previous.entry.planned !== previous.mip) this.cancel();
    if (this.transfer?.cancelled) return 0; // Aborted provider still owns its accounted work until settled.
    if (!this.transfer) {
      const ordered = [...this.entries].sort((a, b) => a.priority - b.priority || a.source.id.localeCompare(b.source.id));
      const surplus = ordered.find(entry => entry.current.baseMipLevel < entry.planned && entry.failure?.mip !== entry.planned);
      if (surplus) this.start(surplus, surplus.planned, false);
      else {
        const candidate = ordered.reverse().find(entry => entry.current.baseMipLevel > entry.planned && entry.failure?.mip !== entry.planned);
        if (candidate) {
          const missing = candidate.bytes[candidate.planned] - candidate.bytes[candidate.current.baseMipLevel];
          if (this.cpuBytes() + missing > this.maxCpuBytes) this.budgetBlocked = true;
          else if (this.backend.imageBytes() + candidate.bytes[candidate.planned] <= this.backend.budgetBytes) this.start(candidate, candidate.planned, false);
          else if (candidate.current !== candidate.source.fallback) this.start(candidate, candidate.source.fallback.baseMipLevel, true);
          else this.budgetBlocked = true;
        }
      }
    }
    const transfer = this.transfer;
    if (!transfer || transfer.loading || !transfer.asset) return 0;
    try {
      if (!transfer.upload) {
        if (this.backend.imageBytes() + transfer.entry.bytes[transfer.mip] > this.backend.budgetBytes) { this.budgetBlocked = true; return 0; }
        transfer.upload = this.backend.stage(transfer.asset);
      }
      const before = transfer.upload.diagnostics().remainingBytes;
      try { transfer.upload.upload(this.maxUploadBytes); }
      finally {
        this.frameUploadedBytes = before - transfer.upload.diagnostics().remainingBytes;
        this.uploadedBytes += this.frameUploadedBytes;
      }
      if (transfer.upload.complete) this.publish(transfer);
    } catch (error) { this.fail(transfer, error); }
    return this.frameUploadedBytes;
  }

  private start(entry: Entry, mip: number, bridge: boolean): void {
    const transfer: Transfer = { entry, mip, bridge, controller: new AbortController(),
      reserved: Math.max(0, entry.bytes[mip] - entry.bytes[entry.current.baseMipLevel]), loading: false, cancelled: false };
    this.transfer = transfer;
    const old = entry.current;
    if (mip >= old.baseMipLevel) {
      transfer.asset = mip === entry.source.fallback.baseMipLevel ? entry.source.fallback :
        { ...old, baseMipLevel: mip, levels: old.levels.slice(mip - old.baseMipLevel) };
      return;
    }
    transfer.loading = true;
    transfer.promise = Promise.resolve().then(async () => {
      const added: PreparedTextureAsset['levels'][number][] = [];
      for (let level = mip; level < old.baseMipLevel; level++) {
        const descriptor = entry.source.mips[level];
        if (transfer.cancelled || this.disposed) return;
        this.inFlightBytes = 2 * descriptor.byteLength; this.requests++;
        const data = await entry.source.provider(descriptor, transfer.controller.signal);
        await verifyTextureStreamMip(descriptor, data, transfer.controller.signal);
        if (transfer.cancelled || this.disposed) return;
        added.push({ width: descriptor.width, height: descriptor.height, data });
        this.inFlightBytes = 0;
      }
      if (!transfer.cancelled && !this.disposed) transfer.asset = { ...old, baseMipLevel: mip, levels: [...added, ...old.levels] };
    }).catch(error => { if (!transfer.cancelled && !this.disposed) this.fail(transfer, error); }).finally(() => {
      this.inFlightBytes = 0; transfer.loading = false;
      if (transfer.cancelled && this.transfer === transfer) this.transfer = undefined;
    });
  }

  private publish(transfer: Transfer): void {
    const entry = transfer.entry, next = transfer.asset!;
    const pin = this.backend.pin(next);
    try { this.backend.replace(entry.current, next); }
    catch (error) { pin.dispose(); throw error; }
    const oldPin = entry.pin;
    if (next.baseMipLevel > entry.current.baseMipLevel) this.evictions++;
    entry.current = next; entry.pin = pin; entry.failure = undefined;
    transfer.upload!.dispose(); oldPin.dispose(); this.transfer = undefined; this.publications++;
  }
  private fail(transfer: Transfer, error: unknown): void {
    transfer.entry.failure = { mip: transfer.mip, message: error instanceof Error ? error.message : String(error) };
    this.cancel();
  }
  private cancel(): void {
    const transfer = this.transfer;
    if (!transfer || transfer.cancelled) return;
    transfer.cancelled = true; this.cancellations++;
    transfer.upload?.dispose(); transfer.upload = undefined;
    transfer.controller.abort();
    if (!transfer.loading) this.transfer = undefined;
  }
  retry(sourceId: string): void {
    this.live(); const entry = this.entries.find(item => item.source.id === sourceId);
    if (!entry) throw new Error('TEXTURE_STREAM_SOURCE_ID');
    entry.failure = undefined;
  }
  /** Wait only for the current IO/hash lifetime, not future render-frame uploads. */
  async idle(): Promise<void> { await this.transfer?.promise; }
  diagnostics() {
    return { sources: this.entries.map(entry => ({ id: entry.source.id, currentMip: entry.current.baseMipLevel,
      wantedMip: entry.wanted, plannedMip: entry.planned, priority: entry.priority, failure: entry.failure ?? null })),
      cpuBytes: this.disposed && !this.transfer ? 0 : this.cpuBytes(), reservedCpuBytes: this.transfer?.reserved ?? 0,
      inFlightBytes: this.inFlightBytes, rootBytes: this.rootBytes, scratchBytes: this.scratchBytes,
      active: this.transfer ? 1 : 0, loading: !!this.transfer?.loading, budgetBlocked: this.budgetBlocked,
      requests: this.requests, cancellations: this.cancellations, publications: this.publications, evictions: this.evictions,
      uploadedBytes: this.uploadedBytes, frameUploadedBytes: this.frameUploadedBytes,
      limits: { maxCpuBytes: this.maxCpuBytes, maxInFlightBytes: this.maxInFlightBytes, maxUploadBytesPerFrame: this.maxUploadBytes } };
  }
  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.disposed = true; this.cancel();
    for (const entry of this.entries) entry.pin?.dispose();
    this.disposal = this.idle().then(() => { this.entries.length = 0; this.identities.clear(); });
    return this.disposal;
  }
}
