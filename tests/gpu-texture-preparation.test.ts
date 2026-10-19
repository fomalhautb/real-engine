import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CompressedTexturePreparation } from '../src/gpu-texture-preparation';

type FakeTexture = { label: string; destroyed: boolean; createView(): { label: string }; destroy(): void };
function texture(label: string): FakeTexture {
  return { label, destroyed: false, createView() { return { label: this.label }; },
    destroy() { this.destroyed = true; } };
}

class FakeDevice {
  adapterInfo = { architecture: 'native', description: 'native GPU' };
  events: string[] = [];
  targets: FakeTexture[] = [];
  failAt: string | null = null;
  private maybeFail(name: string): void {
    if (this.failAt === name) { this.failAt = null; throw new Error(`FAKE_${name.toUpperCase()}_FAILURE`); }
  }
  createBindGroupLayout = () => { this.events.push('layout'); this.maybeFail('layout'); return {}; };
  createShaderModule = () => { this.events.push('shader'); this.maybeFail('shader'); return {}; };
  createPipelineLayout = () => { this.events.push('pipeline-layout'); this.maybeFail('pipeline-layout'); return {}; };
  createRenderPipeline = () => { this.events.push('pipeline'); this.maybeFail('pipeline'); return {}; };
  createSampler = () => { this.events.push('sampler'); this.maybeFail('sampler'); return {}; };
  createTexture = ({ label }: { label: string }) => {
    this.events.push('target'); this.maybeFail('target');
    const target = texture(label); this.targets.push(target); return target;
  };
  createBindGroup = ({ entries }: { entries: { resource: { label?: string } }[] }) => {
    this.events.push(`group:${entries[0].resource.label}`); this.maybeFail('group'); return {};
  };
  createCommandEncoder = () => {
    this.events.push('encoder'); this.maybeFail('encoder');
    return { beginRenderPass: () => {
      this.events.push('begin'); this.maybeFail('begin');
      return { setPipeline: () => this.events.push('setPipeline'),
        setBindGroup: () => this.events.push('setBindGroup'),
        draw: (vertices: number) => this.events.push(`draw:${vertices}`),
        end: () => this.events.push('end') };
    }, finish: () => { this.events.push('finish'); this.maybeFail('finish'); return {}; } };
  };
  queue = { submit: (_commands: unknown[]) => { this.events.push('submit'); this.maybeFail('submit'); } };
}

function ledger() {
  let bytes = 0;
  const changes: number[] = [];
  return { account: (delta: number) => { bytes += delta; changes.push(bytes); },
    get bytes() { return bytes; }, get changes() { return changes; } };
}

beforeEach(() => {
  vi.stubGlobal('GPUShaderStage', { FRAGMENT: 1 });
  vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1 });
});
afterEach(() => vi.unstubAllGlobals());

describe('compressed texture preparation', () => {
  it.each([
    [{ architecture: 'SwiftShader', description: '' }, 'swiftshader-render'],
    [{ architecture: '', description: 'Google SWIFTSHADER device' }, 'swiftshader-render'],
    [{ architecture: 'gen-12lp', description: 'Intel graphics' }, 'none'],
  ] as const)('detects adapter mode from architecture or description', (info, mode) => {
    const device = new FakeDevice(), tracked = ledger();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
      tracked.account, info);
    expect(preparation.diagnostics()).toEqual({ mode, imagesPrepared: 0, ownedGpuBytes: 0 });
    preparation.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('allocates no GPU objects and submits no work on native devices', () => {
    const device = new FakeDevice(), tracked = ledger();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice, tracked.account);
    preparation.prepare(texture('native image 1') as unknown as GPUTexture);
    preparation.prepare(texture('native image 2') as unknown as GPUTexture);
    expect(device.events).toEqual([]);
    expect(device.targets).toEqual([]);
    expect(preparation.diagnostics()).toEqual({ mode: 'none', imagesPrepared: 0, ownedGpuBytes: 0 });
    preparation.dispose(); preparation.dispose();
    expect(tracked.bytes).toBe(0);
    expect(() => preparation.prepare(texture('late') as unknown as GPUTexture)).toThrow(/DISPOSED/);
  });

  it('lazily allocates a 4-byte target once, then reuses one pipeline for multiple images', () => {
    const device = new FakeDevice(), tracked = ledger();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
      tracked.account, { architecture: 'SwiftShader', description: '' });
    expect(device.events).toEqual([]);
    preparation.prepare(texture('image A') as unknown as GPUTexture);
    preparation.prepare(texture('image B') as unknown as GPUTexture);
    expect(device.events.filter(event => event === 'pipeline')).toHaveLength(1);
    expect(device.events.filter(event => event === 'target')).toHaveLength(1);
    expect(device.events.filter(event => event === 'submit')).toHaveLength(2);
    expect(device.events.filter(event => event === 'draw:3')).toHaveLength(2);
    expect(device.events.filter(event => event.startsWith('group:'))).toEqual(['group:image A', 'group:image B']);
    expect(device.targets[0].destroyed).toBe(false);
    expect(preparation.diagnostics()).toEqual({ mode: 'swiftshader-render', imagesPrepared: 2, ownedGpuBytes: 4 });
    expect(tracked.changes).toEqual([4]);
    preparation.dispose(); preparation.dispose();
    expect(device.targets[0].destroyed).toBe(true);
    expect(tracked.changes).toEqual([4, 0]);
  });

  it('submits one draw after the caller uploads all mip levels, in queue order', () => {
    const device = new FakeDevice(), tracked = ledger();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
      tracked.account, { architecture: 'SwiftShader', description: '' });
    const image = texture('uploaded image');
    device.events.push('upload:mip0', 'upload:mip1', 'upload:mip2');
    preparation.prepare(image as unknown as GPUTexture);
    expect(device.events.slice(-9)).toEqual([
      'group:uploaded image', 'encoder', 'begin', 'setPipeline', 'setBindGroup',
      'draw:3', 'end', 'finish', 'submit',
    ]);
    expect(device.events.indexOf('upload:mip2')).toBeLessThan(device.events.indexOf('submit'));
    expect(preparation.diagnostics().imagesPrepared).toBe(1);
    preparation.dispose();
  });

  it.each(['layout', 'shader', 'pipeline-layout', 'pipeline', 'sampler', 'target'])(
    'leaves no owned target or accounted bytes when initial %s creation fails', failure => {
      const device = new FakeDevice(), tracked = ledger();
      const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
        tracked.account, { architecture: 'SwiftShader', description: '' });
      device.failAt = failure;
      expect(() => preparation.prepare(texture('failed') as unknown as GPUTexture))
        .toThrow(`FAKE_${failure.toUpperCase()}_FAILURE`);
      expect(preparation.diagnostics()).toEqual({ mode: 'swiftshader-render', imagesPrepared: 0, ownedGpuBytes: 0 });
      expect(device.targets.every(target => target.destroyed)).toBe(true);
      expect(tracked.bytes).toBe(0);
      preparation.dispose();
      expect(tracked.bytes).toBe(0);
    },
  );

  it('does not count failed queue submission as a prepared image and disposes the shared target', () => {
    const device = new FakeDevice(), tracked = ledger();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
      tracked.account, { architecture: 'SwiftShader', description: '' });
    device.failAt = 'submit';
    expect(() => preparation.prepare(texture('failed') as unknown as GPUTexture)).toThrow('FAKE_SUBMIT_FAILURE');
    expect(preparation.diagnostics().imagesPrepared).toBe(0);
    preparation.dispose();
    expect(device.targets).toHaveLength(1);
    expect(device.targets[0].destroyed).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('destroys the allocated target if its accounting callback rejects ownership', () => {
    const device = new FakeDevice();
    const preparation = new CompressedTexturePreparation(device as unknown as GPUDevice,
      () => { throw new Error('FAKE_ACCOUNT_FAILURE'); },
      { architecture: 'SwiftShader', description: '' });
    expect(() => preparation.prepare(texture('failed') as unknown as GPUTexture)).toThrow('FAKE_ACCOUNT_FAILURE');
    expect(device.targets).toHaveLength(1);
    expect(device.targets[0].destroyed).toBe(true);
    expect(preparation.diagnostics().ownedGpuBytes).toBe(0);
    preparation.dispose();
  });
});
