import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MaterialResources } from '../src/gpu-material';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../src/material';
import type { MaterialBinding } from '../src/contracts';
import type { TextureAsset } from '../src/texture-assets';
import type { PreparedTextureAsset, PreparedTextureFormat } from '../src/texture-storage';

type FakeTexture = { descriptor: { label?: string; format?: string; size?: readonly number[]; mipLevelCount?: number };
  destroyed: boolean; destroy(): void; createView(): { texture: FakeTexture } };
type FakeBuffer = { bytes: Uint8Array; destroyed: boolean; destroy(): void };

class FakeDevice {
  adapterInfo = { architecture: 'native', description: 'native GPU' };
  features = new Set<string>();
  limits = { maxTextureDimension2D: 8192 };
  textures: FakeTexture[] = [];
  buffers: FakeBuffer[] = [];
  samplers: Record<string, unknown>[] = [];
  bindGroups: { entries: { binding: number; resource: unknown }[] }[] = [];
  writes: { texture: FakeTexture; mipLevel: number; origin?: { x: number; y: number; z: number }; data: Uint8Array;
    layout: { bytesPerRow: number; rowsPerImage?: number }; extent: readonly number[] }[] = [];
  bufferWrites: { target: FakeBuffer; offset: number; byteLength: number }[] = [];
  events: string[] = [];
  submissions = 0;
  failSubmit = false;
  failTextureAt = 0;
  failWriteTextureAt = 0;
  failBufferAt = 0;
  failBindGroupAt = 0;
  bindGroupCalls = 0;
  createBindGroupLayout = () => ({});
  createShaderModule = () => ({});
  createPipelineLayout = () => ({});
  createRenderPipeline = () => ({});
  createCommandEncoder = () => ({ beginRenderPass: () => ({ setPipeline: () => {},
    setBindGroup: () => {}, draw: () => this.events.push('draw'), end: () => {} }),
  finish: () => ({}) });
  createSampler = (descriptor: Record<string, unknown>) => {
    this.samplers.push(descriptor);
    return { descriptor };
  };
  createTexture = (descriptor: FakeTexture['descriptor']): FakeTexture => {
    if (this.textures.length + 1 === this.failTextureAt) {
      this.failTextureAt = 0;
      throw new Error('FAKE_TEXTURE_FAILURE');
    }
    const texture: FakeTexture = { descriptor, destroyed: false,
      destroy() { this.destroyed = true; }, createView() { return { texture: this }; } };
    this.textures.push(texture);
    return texture;
  };
  createBuffer = ({ size }: { size: number }): FakeBuffer => {
    if (this.buffers.length + 1 === this.failBufferAt) {
      this.failBufferAt = 0;
      throw new Error('FAKE_BUFFER_FAILURE');
    }
    const buffer: FakeBuffer = { bytes: new Uint8Array(size), destroyed: false,
      destroy() { this.destroyed = true; } };
    this.buffers.push(buffer);
    return buffer;
  };
  createBindGroup = (group: { entries: { binding: number; resource: unknown }[] }) => {
    this.bindGroupCalls++;
    if (this.bindGroupCalls === this.failBindGroupAt) {
      this.failBindGroupAt = 0;
      throw new Error('FAKE_BIND_GROUP_FAILURE');
    }
    this.bindGroups.push(group);
    return group;
  };
  queue = {
    writeTexture: (destination: { texture: FakeTexture; mipLevel?: number; origin?: { x: number; y: number; z: number } }, data: Uint8Array,
      layout: { bytesPerRow: number; rowsPerImage?: number }, extent: readonly number[]) => {
      if (this.writes.length + 1 === this.failWriteTextureAt) {
        this.failWriteTextureAt = 0;
        throw new Error('FAKE_UPLOAD_FAILURE');
      }
      this.writes.push({ texture: destination.texture, mipLevel: destination.mipLevel ?? 0,
        origin: destination.origin,
        data, layout, extent });
      this.events.push(`upload:${destination.mipLevel ?? 0}`);
    },
    writeBuffer: (target: FakeBuffer, offset: number, data: Float32Array) => {
      const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      target.bytes.set(source, offset);
      this.bufferWrites.push({ target, offset, byteLength: source.byteLength });
    },
    submit: (_commands: unknown[]) => {
      this.events.push('submit'); this.submissions++;
      if (this.failSubmit) { this.failSubmit = false; throw new Error('FAKE_PREPARATION_SUBMIT_FAILURE'); }
    },
  };
}

function ledger() {
  let bytes = 0;
  const changes: number[] = [];
  return { account: (delta: number) => { bytes += delta; changes.push(bytes); },
    get bytes() { return bytes; }, get changes() { return changes; } };
}

function prepared(width: number, height: number, format: PreparedTextureFormat = 'bc7', baseMipLevel = 0): PreparedTextureAsset {
  const block = format === 'rgba8' ? 1 : 4;
  const blockBytes = format === 'rgba8' ? 4 : 16;
  const levels: { width: number; height: number; data: Uint8Array }[] = [];
  for (let level = baseMipLevel; ; level++) {
    const w = Math.max(1, Math.floor(width / 2 ** level));
    const h = Math.max(1, Math.floor(height / 2 ** level));
    const data = new Uint8Array(Math.ceil(w / block) * Math.ceil(h / block) * blockBytes);
    data.fill(level + 17);
    levels.push({ width: w, height: h, data });
    if (w === 1 && h === 1) break;
  }
  return { kind: 'prepared', width, height, format, colorSpace: 'srgb', baseMipLevel, levels };
}

function legacy(): TextureAsset {
  return { width: 4, height: 4, colorSpace: 'linear', pixels: new Uint8Array(64).fill(150) };
}

const textured = { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.texture(0) };
const parametrized = { ...DEFAULT_MATERIAL_DEFINITION,
  baseColor: m.multiply(m.texture(0), m.parameter(0)) };
function binding(asset: PreparedTextureAsset | TextureAsset, extras: Partial<MaterialBinding> = {}): MaterialBinding {
  return { definition: textured, textures: [asset], ...extras };
}

beforeEach(() => {
  vi.stubGlobal('GPUShaderStage', { FRAGMENT: 1, COMPUTE: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 });
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
});
afterEach(() => vi.unstubAllGlobals());

describe('material texture residency and ownership', () => {
  it.each([
    ['bc7', 'texture-compression-bc'],
    ['etc2-rgba8', 'texture-compression-etc2'],
    ['astc-4x4', 'texture-compression-astc'],
  ] as const)('rejects unavailable %s before allocating its texture', (format, feature) => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    expect(() => resources.create(binding(prepared(4, 4, format)))).toThrow(feature);
    expect(device.textures).toHaveLength(1); // Constructor fallback only.
    expect(device.buffers).toHaveLength(0);
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(tracked.bytes).toBe(4);
    resources.dispose();
    expect(device.textures[0].destroyed).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('uses native compressed bytes and requires the old image to fit alongside its replacement', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account, 150);
    const compressed = resources.create(binding(prepared(8, 8)));
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 112 });
    expect(tracked.bytes).toBe(4 + 112 + 320);
    expect(() => resources.create(binding(legacy()))).toThrow(/MATERIAL_TEXTURE_BUDGET/);
    expect(device.textures).toHaveLength(2);
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 112 });
    resources.release(compressed);
    const replacement = resources.create(binding(legacy()));
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 84 });
    resources.release(replacement);
    resources.dispose();
    expect(tracked.bytes).toBe(0);
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.changes.every(bytes => bytes >= 0)).toBe(true);
  });

  it('shares one texture by asset identity and releases it after the last material', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(4, 4), first = resources.create(binding(asset)), second = resources.create(binding(asset));
    const image = device.textures[1];
    expect(device.textures).toHaveLength(2);
    expect(resources.diagnostics()).toEqual({ materials: 2, images: 1, imageBytes: 48 });
    resources.release(first);
    expect(image.destroyed).toBe(false);
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 48 });
    resources.release(first); // Idempotent.
    resources.release(second);
    expect(image.destroyed).toBe(true);
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    resources.dispose(); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it.each(['texture', 'upload', 'buffer', 'bindGroup'] as const)(
    'cleans all acquired resources when %s creation fails', failure => {
      const device = new FakeDevice(), tracked = ledger();
      device.features.add('texture-compression-bc');
      const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
      if (failure === 'texture') device.failTextureAt = 2;
      if (failure === 'upload') device.failWriteTextureAt = 3; // fallback, mip 0, then mip 1.
      if (failure === 'buffer') device.failBufferAt = 1;
      if (failure === 'bindGroup') device.failBindGroupAt = 1;
      expect(() => resources.create(binding(prepared(8, 8)))).toThrow(/FAKE_.*_FAILURE/);
      expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
      expect(tracked.bytes).toBe(4);
      expect(device.textures.slice(1).every(texture => texture.destroyed)).toBe(true);
      expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
      const recovered = resources.create(binding(prepared(4, 4)));
      resources.release(recovered); resources.dispose();
      expect(tracked.bytes).toBe(0);
    },
  );

  it('uploads prepared payloads by identity with exact block-padded extents', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(8, 8), material = resources.create(binding(asset));
    const writes = device.writes.slice(1);
    expect(device.textures[1].descriptor).toMatchObject({ size: [8, 8], mipLevelCount: 4,
      format: 'bc7-rgba-unorm-srgb' });
    expect(writes.map(write => [write.mipLevel, write.layout.bytesPerRow, write.layout.rowsPerImage,
      write.extent])).toEqual([
      [0, 32, 2, [8, 8]], [1, 16, 1, [4, 4]],
      [2, 16, 1, [4, 4]], [3, 16, 1, [4, 4]],
    ]);
    writes.forEach((write, index) => expect(write.data).toBe(asset.levels[index].data));
    resources.release(material); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('translates source-space sampler clamps and preserves residency metadata on parameter updates', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(11, 9, 'rgba8', 1);
    const material = resources.create(binding(asset, { definition: parametrized,
      parameters: new Float32Array([0.1, 0.2, 0.3, 0.4]),
      samplers: [{ lodMinClamp: 0.5, lodMaxClamp: 6, minFilter: 'linear' }] }));
    const sampler = device.samplers.at(-1)!;
    expect(sampler.lodMinClamp).toBe(0);
    expect(sampler.lodMaxClamp).toBe(5);
    const stored = new Float32Array(device.buffers[0].bytes.buffer);
    expect(Array.from(stored.slice(64, 68))).toEqual([1, Math.fround(1.1), Math.fround(1.125), 0]);
    const before = Array.from(stored.slice(64));
    resources.updateParameters(material, new Float32Array([0.8, 0.7, 0.6, 0.5]));
    expect(device.bufferWrites.at(-1)).toMatchObject({ target: device.buffers[0], offset: 0, byteLength: 256 });
    expect(Array.from(stored.slice(0, 4))).toEqual(Array.from(new Float32Array([0.8, 0.7, 0.6, 0.5])));
    expect(Array.from(stored.slice(64))).toEqual(before);
    resources.release(material); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('prepares one shared compressed image only after all mip uploads on SwiftShader', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'SwiftShader', description: 'software adapter' };
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(8, 8);
    const first = resources.create(binding(asset)), second = resources.create(binding(asset));
    expect(resources.preparationDiagnostics()).toEqual({ mode: 'swiftshader-render',
      imagesPrepared: 1, ownedGpuBytes: 4 });
    expect(device.submissions).toBe(1);
    expect(device.textures.filter(texture => texture.descriptor.label === 'compressed image preparation target')).toHaveLength(1);
    expect(device.events.slice(1, 7)).toEqual(['upload:0', 'upload:1', 'upload:2', 'upload:3', 'draw', 'submit']);
    expect(tracked.bytes).toBe(4 + 112 + 4 + 2 * 320);
    resources.release(first); resources.release(second); resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('does not prepare RGBA8 images even when SwiftShader mode is active', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'swiftshader', description: '' };
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const one = resources.create(binding(prepared(4, 4, 'rgba8')));
    const two = resources.create(binding(legacy()));
    expect(resources.preparationDiagnostics()).toEqual({ mode: 'swiftshader-render',
      imagesPrepared: 0, ownedGpuBytes: 0 });
    expect(device.submissions).toBe(0);
    expect(device.textures.filter(texture => texture.descriptor.label === 'compressed image preparation target')).toHaveLength(0);
    resources.release(one); resources.release(two); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('does not publish a compressed image when preparation submission fails', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'SwiftShader', description: '' };
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    device.failSubmit = true;
    expect(() => resources.create(binding(prepared(4, 4)))).toThrow('FAKE_PREPARATION_SUBMIT_FAILURE');
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(resources.preparationDiagnostics().imagesPrepared).toBe(0);
    expect(device.textures[1].destroyed).toBe(true);
    expect(device.buffers).toHaveLength(0);
    resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('atomically replaces shared and repeated slots while retaining handles, samplers and hot parameters', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const source = legacy(), replacement = prepared(8, 8, 'rgba8', 1), unrelatedAsset = legacy();
    const first = resources.create(binding(source, { definition: parametrized,
      parameters: new Float32Array([0.1, 0.2, 0.3, 0.4]) }));
    const duplicate = resources.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.add(m.texture(0), m.texture(1)) }, textures: [source, source],
    samplers: [{ magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest',
      lodMinClamp: 1, lodMaxClamp: 3 }, { lodMinClamp: 2, lodMaxClamp: 5 }] });
    const unrelated = resources.create(binding(unrelatedAsset));
    resources.updateParameters(first, new Float32Array([0.8, 0.7, 0.6, 0.5]));
    const oldSource = first.images[0].texture, oldFirstUniform = first.uniform;
    const oldDuplicateUniform = duplicate.uniform, oldFirstProgram = first.program;
    const oldDuplicateProgram = duplicate.program;
    const oldUnrelated = { group: unrelated.bindGroup, uniform: unrelated.uniform,
      image: unrelated.images[0].texture };
    expect(resources.replaceTexture(source, replacement)).toEqual({ materials: 2, slots: 3 });
    expect(first.program).toBe(oldFirstProgram);
    expect(duplicate.program).toBe(oldDuplicateProgram);
    expect(first.disposed).toBe(false);
    expect(duplicate.disposed).toBe(false);
    expect(first.uniform).not.toBe(oldFirstUniform);
    expect(duplicate.uniform).not.toBe(oldDuplicateUniform);
    expect((oldFirstUniform as unknown as FakeBuffer).destroyed).toBe(true);
    expect((oldDuplicateUniform as unknown as FakeBuffer).destroyed).toBe(true);
    expect((oldSource as unknown as FakeTexture).destroyed).toBe(true);
    expect(first.images[0].texture).toBe(duplicate.images[0].texture);
    expect(duplicate.images[0].texture).toBe(duplicate.images[1].texture);
    expect(unrelated.bindGroup).toBe(oldUnrelated.group);
    expect(unrelated.uniform).toBe(oldUnrelated.uniform);
    expect(unrelated.images[0].texture).toBe(oldUnrelated.image);
    const newTexture = first.images[0].texture as unknown as FakeTexture;
    expect(device.writes.filter(write => write.texture === newTexture)).toHaveLength(3);
    const firstValues = new Float32Array((first.uniform as unknown as FakeBuffer).bytes.buffer);
    expect(Array.from(firstValues.slice(0, 4))).toEqual(Array.from(new Float32Array([0.8, 0.7, 0.6, 0.5])));
    expect(firstValues[64]).toBe(1);
    const group = duplicate.bindGroup as unknown as { entries: { binding: number; resource: { descriptor?: GPUSamplerDescriptor } }[] };
    const slot0 = group.entries.find(entry => entry.binding === 5)!.resource.descriptor!;
    const slot1 = group.entries.find(entry => entry.binding === 6)!.resource.descriptor!;
    expect([slot0.minFilter, slot0.lodMinClamp, slot0.lodMaxClamp]).toEqual(['nearest', 0, 2]);
    expect([slot1.minFilter, slot1.lodMinClamp, slot1.lodMaxClamp]).toEqual(['linear', 1, 4]);
    expect(resources.diagnostics()).toEqual({ materials: 3, images: 2, imageBytes: 168 });
    resources.release(first); resources.release(duplicate); resources.release(unrelated); resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('does not allocate for self replacement or an unreferenced source', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const source = legacy(), target = prepared(4, 4, 'rgba8');
    const material = resources.create(binding(source));
    const before = { textures: device.textures.length, buffers: device.buffers.length,
      groups: device.bindGroups.length, writes: device.writes.length, bytes: tracked.bytes,
    group: material.bindGroup, uniform: material.uniform };
    expect(resources.replaceTexture(source, source)).toEqual({ materials: 0, slots: 0 });
    expect(resources.replaceTexture(legacy(), target)).toEqual({ materials: 0, slots: 0 });
    expect({ textures: device.textures.length, buffers: device.buffers.length,
      groups: device.bindGroups.length, writes: device.writes.length, bytes: tracked.bytes,
      group: material.bindGroup, uniform: material.uniform }).toEqual(before);
    resources.release(material); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('rolls back every binding when the second staged bind group fails', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const source = legacy(), replacement = prepared(4, 4, 'rgba8');
    const first = resources.create(binding(source)), second = resources.create(binding(source));
    const old = [first, second].map(material => ({ uniform: material.uniform,
      group: material.bindGroup, images: material.images.slice(), program: material.program }));
    const bytesBefore = tracked.bytes, texturesBefore = device.textures.length;
    const buffersBefore = device.buffers.length;
    device.failBindGroupAt = device.bindGroupCalls + 2;
    expect(() => resources.replaceTexture(source, replacement)).toThrow('FAKE_BIND_GROUP_FAILURE');
    for (let index = 0; index < 2; index++) {
      const material = [first, second][index], previous = old[index];
      expect(material.uniform).toBe(previous.uniform);
      expect(material.bindGroup).toBe(previous.group);
      expect(material.images).toEqual(previous.images);
      expect(material.program).toBe(previous.program);
      expect((material.uniform as unknown as FakeBuffer).destroyed).toBe(false);
      expect((material.images[0].texture as unknown as FakeTexture).destroyed).toBe(false);
    }
    expect(device.buffers.slice(buffersBefore).every(buffer => buffer.destroyed)).toBe(true);
    expect(device.textures.slice(texturesBefore).every(texture => texture.destroyed)).toBe(true);
    expect(resources.diagnostics()).toEqual({ materials: 2, images: 1, imageBytes: 84 });
    expect(tracked.bytes).toBe(bytesBefore);
    expect(resources.replaceTexture(source, replacement)).toEqual({ materials: 2, slots: 2 });
    resources.release(first); resources.release(second); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('requires simultaneous old and new image budget, then uploads a shared replacement once', () => {
    const source = legacy(), replacement = prepared(4, 4, 'rgba8');
    const limitedDevice = new FakeDevice(), limitedLedger = ledger();
    const limited = new MaterialResources(limitedDevice as unknown as GPUDevice, limitedLedger.account, 167);
    const old = limited.create(binding(source));
    const oldGroup = old.bindGroup, oldUniform = old.uniform;
    expect(() => limited.replaceTexture(source, replacement)).toThrow(/MATERIAL_TEXTURE_BUDGET/);
    expect(old.bindGroup).toBe(oldGroup); expect(old.uniform).toBe(oldUniform);
    expect(limitedDevice.textures).toHaveLength(2);
    expect(limited.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 84 });
    limited.release(old); limited.dispose();
    expect(limitedLedger.bytes).toBe(0);

    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account, 168);
    const first = resources.create(binding(source)), second = resources.create(binding(source));
    expect(resources.replaceTexture(source, replacement)).toEqual({ materials: 2, slots: 2 });
    expect(first.images[0].texture).toBe(second.images[0].texture);
    const image = first.images[0].texture as unknown as FakeTexture;
    expect(device.writes.filter(write => write.texture === image)).toHaveLength(3);
    expect(resources.diagnostics()).toEqual({ materials: 2, images: 1, imageBytes: 84 });
    resources.release(first); resources.release(second); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('rejects foreign-owner parameter updates and releases without touching either owner', () => {
    const device = new FakeDevice(), trackedA = ledger(), trackedB = ledger();
    const a = new MaterialResources(device as unknown as GPUDevice, trackedA.account);
    const b = new MaterialResources(device as unknown as GPUDevice, trackedB.account);
    const material = a.create(binding(legacy()));
    const original = new Uint8Array((material.uniform as unknown as FakeBuffer).bytes);
    expect(() => b.updateParameters(material, new Float32Array())).toThrow('MATERIAL_NOT_OWNED');
    expect(() => b.release(material)).toThrow('MATERIAL_NOT_OWNED');
    expect(material.disposed).toBe(false);
    expect((material.uniform as unknown as FakeBuffer).bytes).toEqual(original);
    expect(a.diagnostics().materials).toBe(1);
    expect(b.diagnostics().materials).toBe(0);
    a.release(material); a.dispose(); b.dispose();
    expect(trackedA.bytes).toBe(0);
    expect(trackedB.bytes).toBe(0);
  });

  it('keeps partial staged images invisible, then atomically publishes the completed upload without rewriting it', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account, 500);
    const source = legacy(), replacement = prepared(8, 8, 'rgba8');
    const first = resources.create(binding(source)), second = resources.create(binding(source));
    const oldGroups = [first.bindGroup, second.bindGroup], oldUniforms = [first.uniform, second.uniform];
    const oldTexture = first.images[0].texture;
    const handle = resources.stageTexture(replacement);
    expect(handle.complete).toBe(false);
    expect(handle.diagnostics()).toEqual({ complete: false, remainingBytes: 340,
      nextRowBytes: 32, allocationBytes: 340 });
    expect(resources.uploadDiagnostics()).toEqual({ active: 1, pending: 1, pendingImageBytes: 340 });
    expect(resources.diagnostics()).toEqual({ materials: 2, images: 1, imageBytes: 424 });
    expect(handle.upload(1)).toBe(0); // Smaller than one block row cannot make partial progress.
    expect(handle.upload(32)).toBe(32);
    const staged = device.textures[2];
    expect(device.writes.filter(write => write.texture === staged)).toHaveLength(1);
    expect(() => resources.create(binding(replacement))).toThrow('MATERIAL_TEXTURE_PENDING_UPLOAD');
    expect(() => resources.replaceTexture(source, replacement)).toThrow('MATERIAL_TEXTURE_PENDING_UPLOAD');
    expect([first.bindGroup, second.bindGroup]).toEqual(oldGroups);
    expect([first.uniform, second.uniform]).toEqual(oldUniforms);
    expect(first.images[0].texture).toBe(oldTexture);
    expect(second.images[0].texture).toBe(oldTexture);
    let uploaded = 32;
    for (let tick = 0; tick < 20 && !handle.complete; tick++) uploaded += handle.upload(64);
    expect(handle.complete).toBe(true);
    expect(uploaded).toBe(340);
    expect(handle.diagnostics()).toEqual({ complete: true, remainingBytes: 0,
      nextRowBytes: 0, allocationBytes: 340 });
    expect(resources.uploadDiagnostics()).toEqual({ active: 1, pending: 0, pendingImageBytes: 0 });
    const writes = device.writes.filter(write => write.texture === staged).length;
    expect(resources.replaceTexture(source, replacement)).toEqual({ materials: 2, slots: 2 });
    expect(device.writes.filter(write => write.texture === staged)).toHaveLength(writes);
    expect(first.images[0].texture).toBe(staged);
    expect(second.images[0].texture).toBe(staged);
    expect([first.bindGroup, second.bindGroup]).not.toEqual(oldGroups);
    expect((oldTexture as unknown as FakeTexture).destroyed).toBe(true);
    handle.dispose();
    expect(staged.destroyed).toBe(false); // Both published materials still reference it.
    resources.release(first); resources.release(second);
    expect(staged.destroyed).toBe(true);
    resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('cancels a partial stage, rejects duplicate pending assets, and can retry the same asset', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(8, 8, 'rgba8');
    const first = resources.stageTexture(asset);
    expect(() => resources.stageTexture(asset)).toThrow('MATERIAL_TEXTURE_UPLOAD_IN_PROGRESS');
    expect(first.upload(32)).toBe(32);
    const cancelledTexture = device.textures[1];
    first.dispose(); first.dispose();
    expect(first.disposed).toBe(true);
    expect(() => first.upload(32)).toThrow('MATERIAL_TEXTURE_UPLOAD_DISPOSED');
    expect(cancelledTexture.destroyed).toBe(true);
    expect(resources.uploadDiagnostics()).toEqual({ active: 0, pending: 0, pendingImageBytes: 0 });
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(tracked.bytes).toBe(4);
    const retry = resources.stageTexture(asset);
    expect(device.textures[2]).not.toBe(cancelledTexture);
    expect(retry.upload(340)).toBe(340);
    expect(retry.complete).toBe(true);
    retry.dispose(); resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('retries a failed row write without advancing or leaking its reserved allocation', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const handle = resources.stageTexture(prepared(8, 8, 'rgba8'));
    expect(handle.upload(32)).toBe(32);
    device.failWriteTextureAt = device.writes.length + 1;
    expect(() => handle.upload(32)).toThrow('FAKE_UPLOAD_FAILURE');
    expect(handle.diagnostics()).toMatchObject({ complete: false, remainingBytes: 308, nextRowBytes: 32 });
    expect(resources.uploadDiagnostics().pendingImageBytes).toBe(340);
    expect(tracked.bytes).toBe(344);
    expect(handle.upload(32)).toBe(32);
    const staged = device.textures[1];
    expect(device.writes.filter(write => write.texture === staged).map(write => write.origin?.y)).toEqual([0, 1]);
    let total = 64;
    for (let tick = 0; tick < 20 && !handle.complete; tick++) total += handle.upload(64);
    expect(total).toBe(340);
    handle.dispose(); resources.dispose();
    expect(staged.destroyed).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('reserves simultaneous stages against the same budget and enforces the active-handle limit', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account, 168);
    const first = resources.stageTexture(prepared(4, 4, 'rgba8'));
    const second = resources.stageTexture(prepared(4, 4, 'rgba8'));
    expect(resources.uploadDiagnostics()).toEqual({ active: 2, pending: 2, pendingImageBytes: 168 });
    expect(tracked.bytes).toBe(172);
    const texturesBefore = device.textures.length;
    expect(() => resources.stageTexture(prepared(4, 4, 'rgba8'))).toThrow('MATERIAL_TEXTURE_BUDGET');
    expect(device.textures).toHaveLength(texturesBefore);
    first.dispose(); second.dispose(); resources.dispose();
    expect(tracked.bytes).toBe(0);

    const manyDevice = new FakeDevice(), manyLedger = ledger();
    const many = new MaterialResources(manyDevice as unknown as GPUDevice, manyLedger.account);
    const asset = prepared(4, 4, 'rgba8');
    const base = many.stageTexture(asset); base.upload(84);
    const pins = [base, ...Array.from({ length: 63 }, () => many.stageTexture(asset))];
    expect(many.uploadDiagnostics()).toEqual({ active: 64, pending: 0, pendingImageBytes: 0 });
    expect(() => many.stageTexture(asset)).toThrow('MATERIAL_TEXTURE_UPLOAD_LIMIT');
    pins.forEach(pin => pin.dispose()); many.dispose();
    expect(manyLedger.bytes).toBe(0);
  });

  it('manager disposal releases unfinished stages and completed pins after material references', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const pending = resources.stageTexture(prepared(8, 8, 'rgba8'));
    pending.upload(32);
    const completeAsset = prepared(4, 4, 'rgba8');
    const complete = resources.stageTexture(completeAsset);
    complete.upload(84);
    const published = resources.create(binding(completeAsset));
    resources.dispose(); resources.dispose();
    expect(pending.disposed).toBe(true);
    expect(complete.disposed).toBe(true);
    expect(published.disposed).toBe(true);
    expect(resources.uploadDiagnostics()).toEqual({ active: 0, pending: 0, pendingImageBytes: 0 });
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });
});
