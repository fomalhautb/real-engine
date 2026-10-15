import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MeshData } from '../src/geometry';

const options = { maxVertices: 8, maxTangentBytes: 1024 * 1024 };

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  for (const [vertex, x, y] of [[0, 0, 0], [1, 1, 0], [2, 0, 1]]) {
    view.setFloat32(vertex * 80, x, true);
    view.setFloat32(vertex * 80 + 4, y, true);
    view.setFloat32(vertex * 80 + 24, 1, true);
  }
  return { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    bounds: { min: [0, 0, 0], max: [1, 1, 0] } };
}

describe('glTF tangent WASM lifecycle', () => {
  beforeEach(() => { vi.resetModules(); });
  afterEach(() => { vi.doUnmock('meshoptimizer/tangents'); });

  it('requires explicit preparation before synchronous generation', async () => {
    const generateTangents = vi.fn();
    vi.doMock('meshoptimizer/tangents', () => ({ MeshoptTangents: {
      supported: true, ready: Promise.resolve(), generateTangents,
    } }));
    const { generateGltfTangents } = await import('../src/gltf-tangents');
    expect(() => generateGltfTangents(triangle(), [], options))
      .toThrow(/GLTF_TANGENT_GENERATOR_NOT_READY/);
    expect(generateTangents).not.toHaveBeenCalled();
  });

  it('shares one pending preparation promise across concurrent callers', async () => {
    let release!: () => void;
    const ready = new Promise<void>(resolve => { release = resolve; });
    const factory = vi.fn(() => ({ MeshoptTangents: {
      supported: true, ready,
      generateTangents: vi.fn(() => new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])),
    } }));
    vi.doMock('meshoptimizer/tangents', factory);
    const { prepareGltfTangents, generateGltfTangents } = await import('../src/gltf-tangents');
    const first = prepareGltfTangents(), second = prepareGltfTangents();
    expect(second).toBe(first);
    release();
    await Promise.all([first, second]);
    expect(prepareGltfTangents()).toBe(first);
    expect(factory).toHaveBeenCalledTimes(1);
    expect(generateGltfTangents(triangle(), [], options).mesh.tangents).toHaveLength(12);
  });

  it('reports unsupported WebAssembly initialization and keeps the failure', async () => {
    vi.doMock('meshoptimizer/tangents', () => ({ MeshoptTangents: {
      supported: false, ready: Promise.resolve(), generateTangents: vi.fn(),
    } }));
    const { prepareGltfTangents, generateGltfTangents } = await import('../src/gltf-tangents');
    await expect(prepareGltfTangents()).rejects.toThrow(/GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED/);
    await expect(prepareGltfTangents()).rejects.toThrow(/GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED/);
    expect(() => generateGltfTangents(triangle(), [], options))
      .toThrow(/GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED/);
  });

  it('reports a rejected WASM ready promise and keeps the failure', async () => {
    vi.doMock('meshoptimizer/tangents', () => ({ MeshoptTangents: {
      supported: true, ready: Promise.reject(new Error('WASM compile failed')), generateTangents: vi.fn(),
    } }));
    const { prepareGltfTangents, generateGltfTangents } = await import('../src/gltf-tangents');
    await expect(prepareGltfTangents()).rejects.toThrow(/GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED/);
    expect(() => generateGltfTangents(triangle(), [], options))
      .toThrow(/GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED/);
  });

  it('does not reuse an uncertain WASM arena after the generator traps', async () => {
    const generateTangents = vi.fn(() => { throw new Error('WASM trap'); });
    vi.doMock('meshoptimizer/tangents', () => ({ MeshoptTangents: {
      supported: true, ready: Promise.resolve(), generateTangents,
    } }));
    const { prepareGltfTangents, generateGltfTangents } = await import('../src/gltf-tangents');
    await prepareGltfTangents();
    expect(() => generateGltfTangents(triangle(), [], options)).toThrow(/GLTF_TANGENT_GENERATION_FAILED/);
    await expect(prepareGltfTangents()).rejects.toThrow(/GLTF_TANGENT_GENERATION_FAILED/);
    expect(() => generateGltfTangents(triangle(), [], options)).toThrow(/GLTF_TANGENT_GENERATION_FAILED/);
    expect(generateTangents).toHaveBeenCalledTimes(1);
  });
});
