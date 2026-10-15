import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadGltf } from '../src/gltf-load';

const tangentPreparation = vi.hoisted(() => vi.fn<() => Promise<void>>());
vi.mock('../src/gltf-tangents', async importOriginal => ({
  ...await importOriginal<typeof import('../src/gltf-tangents')>(),
  prepareGltfTangents: tangentPreparation,
}));

const rootUrl = 'https://assets.example.test/scenes/SimpleSkin.gltf';
const directory = new URL('../public/assets/gltf/SimpleSkin/glTF/', import.meta.url);
const names = [
  'SimpleSkin.gltf', 'SimpleSkin_geometry.bin', 'SimpleSkin_skinningData.bin',
  'SimpleSkin_inverseBindMatrices.bin', 'SimpleSkin_animation.bin',
] as const;
const files = new Map<string, Uint8Array>(names.map(name => [name, Uint8Array.from(readFileSync(new URL(name, directory)))]));
interface FixtureJson {
  buffers: Array<{ uri?: string; byteLength: number }>;
  bufferViews: Array<{ buffer: number; byteOffset?: number; byteLength: number }>;
  images?: unknown[];
  textures?: unknown[];
}
const sourceJson = () => JSON.parse(new TextDecoder().decode(files.get('SimpleSkin.gltf'))) as FixtureJson;

function response(bytes: Uint8Array, headers: HeadersInit = {}): Response {
  return new Response(new Uint8Array(bytes), { headers });
}

function fixtureFetch(requests: URL[], root: Uint8Array = files.get('SimpleSkin.gltf')!): typeof fetch {
  return async (input, init) => {
    const url = new URL(String(input));
    requests.push(url);
    expect(init?.credentials).toBe('omit');
    const name = url.pathname.split('/').at(-1)!;
    const bytes = name === 'SimpleSkin.gltf' ? root : files.get(name);
    if (!bytes) return new Response('missing fixture', { status: 404 });
    return response(bytes);
  };
}

function glbWithExternalBuffers(): Uint8Array {
  const json = sourceJson();
  delete json.buffers[0].uri; // GLB BIN chunk replaces the first external buffer.
  const rawJson = new TextEncoder().encode(JSON.stringify(json));
  const paddedJson = new Uint8Array(Math.ceil(rawJson.length / 4) * 4).fill(0x20);
  paddedJson.set(rawJson);
  const bin = files.get('SimpleSkin_geometry.bin')!;
  const bytes = new Uint8Array(12 + 8 + paddedJson.length + 8 + bin.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, bytes.length, true);
  view.setUint32(12, paddedJson.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.set(paddedJson, 20);
  const binHeader = 20 + paddedJson.length;
  view.setUint32(binHeader, bin.length, true);
  view.setUint32(binHeader + 4, 0x004e4942, true);
  bytes.set(bin, binHeader + 8);
  return bytes;
}

describe('loadGltf bounded resource loading', () => {
  afterEach(() => { vi.unstubAllGlobals(); tangentPreparation.mockReset(); });
  it('resolves each Khronos buffer relative to the document and imports its skin', async () => {
    const requests: URL[] = [];
    const signal = new AbortController().signal;
    const fetcher = fixtureFetch(requests);
    const asset = await loadGltf(rootUrl, { fetch: fetcher, signal });
    expect(requests.map(url => url.href)).toEqual(names.map(name => `https://assets.example.test/scenes/${name}`));
    expect(asset.nodes).toHaveLength(3);
    expect(asset.skins[0].joints).toEqual([1, 2]);
    expect(asset.meshes[0].primitives[0].mesh.vertexCount).toBe(24);
    expect(asset.rig.clips[0].tracks[0].times).toHaveLength(12);
    expect(tangentPreparation).not.toHaveBeenCalled();
  });

  it('awaits tangent preparation only for a normal-mapped primitive needing generation and honors abort afterward', async () => {
    const json = sourceJson() as FixtureJson & {
      materials: Array<Record<string, unknown>>;
      meshes: Array<{ primitives: Array<Record<string, unknown>> }>;
    };
    json.materials = [{ normalTexture: { index: 0 } }];
    json.meshes[0].primitives[0].material = 0;
    const requests: URL[] = [];
    const controller = new AbortController();
    tangentPreparation.mockImplementationOnce(async () => { controller.abort(); });
    await expect(loadGltf(rootUrl, {
      fetch: fixtureFetch(requests, new TextEncoder().encode(JSON.stringify(json))), signal: controller.signal,
    })).rejects.toMatchObject({ name: 'AbortError' });
    expect(tangentPreparation).toHaveBeenCalledOnce();
    expect(requests.map(url => url.pathname.split('/').at(-1))).toEqual(['SimpleSkin.gltf']);
  });

  it('skips tangent preparation when a normal-mapped primitive declares both source streams', async () => {
    const json = sourceJson() as FixtureJson & {
      materials: Array<Record<string, unknown>>;
      meshes: Array<{ primitives: Array<{ attributes: Record<string, number>; material?: number }> }>;
    };
    json.materials = [{ normalTexture: { index: 0 } }];
    const primitive = json.meshes[0].primitives[0];
    primitive.material = 0;
    primitive.attributes.NORMAL = primitive.attributes.POSITION;
    primitive.attributes.TANGENT = primitive.attributes.POSITION;
    await expect(loadGltf(rootUrl, {
      fetch: fixtureFetch([], new TextEncoder().encode(JSON.stringify(json))),
    })).rejects.toThrow(/GLTF_INDEX: textureInfo.index/);
    expect(tangentPreparation).not.toHaveBeenCalled();
  });

  it('prepares and generates tangents while loading a normal-mapped triangle without authored TANGENT', async () => {
    const realTangents = await vi.importActual<typeof import('../src/gltf-tangents')>('../src/gltf-tangents');
    tangentPreparation.mockImplementationOnce(() => realTangents.prepareGltfTangents());
    const geometry = new Float32Array(24);
    geometry.set([0, 0, 0, 1, 0, 0, 0, 1, 0], 0);
    geometry.set([0, 0, 1, 0, 0, 1, 0, 0, 1], 9);
    geometry.set([0, 0, 1, 0, 0, 1], 18);
    const json = {
      asset: { version: '2.0' },
      buffers: [{ uri: 'triangle.bin', byteLength: geometry.byteLength }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36 },
        { buffer: 0, byteOffset: 36, byteLength: 36 },
        { buffer: 0, byteOffset: 72, byteLength: 24 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      images: [{ uri: 'normal.png', mimeType: 'image/png' }], textures: [{ source: 0 }],
      materials: [{ normalTexture: { index: 0 } }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, material: 0 }] }],
      nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }],
    };
    const close = vi.fn();
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 1, height: 1, close }));
    vi.stubGlobal('OffscreenCanvas', class {
      getContext() { return { drawImage: () => {}, getImageData: () => ({ data: new Uint8ClampedArray([128, 128, 255, 255]) }) }; }
    });
    const root = new TextEncoder().encode(JSON.stringify(json));
    const fetcher: typeof fetch = async input => {
      const name = new URL(String(input)).pathname.split('/').at(-1);
      if (name === 'SimpleSkin.gltf') return response(root);
      if (name === 'triangle.bin') return response(new Uint8Array(geometry.buffer));
      if (name === 'normal.png') return response(new Uint8Array([137, 80, 78, 71]));
      return new Response('missing', { status: 404 });
    };
    const asset = await loadGltf(rootUrl, { fetch: fetcher });
    const mesh = asset.meshes[0].primitives[0].mesh;
    expect(tangentPreparation).toHaveBeenCalledOnce();
    expect(mesh.tangents).toHaveLength(mesh.vertexCount * 4);
    expect(mesh.tangents![0]).toBeCloseTo(1, 5);
    expect(mesh.tangents![1]).toBeCloseTo(0, 5);
    expect(asset.warnings.some(warning => /UV-derivative frame fallback/.test(warning))).toBe(false);
    expect(close).toHaveBeenCalledOnce();
  });

  it('loads a data URI buffer through the same bounded fetch contract', async () => {
    const json = sourceJson();
    json.buffers[0].uri = `data:application/octet-stream;base64,${Buffer.from(files.get('SimpleSkin_geometry.bin')!).toString('base64')}`;
    const modified = new TextEncoder().encode(JSON.stringify(json));
    const requests: URL[] = [];
    const baseFetch = fixtureFetch(requests, modified);
    const fetcher: typeof fetch = async (input, init) => {
      const url = new URL(String(input));
      if (url.protocol === 'data:') {
        requests.push(url);
        return response(Uint8Array.from(Buffer.from(url.href.split(',')[1], 'base64')));
      }
      return baseFetch(input, init);
    };
    const asset = await loadGltf(rootUrl, { fetch: fetcher });
    expect(requests[1].protocol).toBe('data:');
    expect(asset.skins[0].joints).toEqual([1, 2]);
  });

  it('uses a GLB BIN chunk for buffer zero and resolves remaining external buffers', async () => {
    const requests: URL[] = [];
    const packed = glbWithExternalBuffers();
    const fetcher: typeof fetch = async (input, init) => {
      const url = new URL(String(input));
      if (url.pathname.endsWith('.glb')) {
        requests.push(url);
        expect(init?.credentials).toBe('omit');
        return response(packed);
      }
      return fixtureFetch(requests)(input, init);
    };
    const asset = await loadGltf('https://assets.example.test/scenes/SimpleSkin.glb', { fetch: fetcher });
    expect(requests.map(url => url.pathname.split('/').at(-1))).toEqual([
      'SimpleSkin.glb', 'SimpleSkin_skinningData.bin', 'SimpleSkin_inverseBindMatrices.bin', 'SimpleSkin_animation.bin',
    ]);
    expect(asset.meshes[0].primitives[0].mesh.indices).toHaveLength(24);
  });

  it('rejects a cumulative Content-Length over budget before consuming another body', async () => {
    const cancel = vi.fn();
    const requests: URL[] = [];
    const fetcher: typeof fetch = async (input) => {
      const url = new URL(String(input));
      requests.push(url);
      if (url.pathname.endsWith('.gltf')) return response(files.get('SimpleSkin.gltf')!);
      const body = new ReadableStream<Uint8Array>({ cancel });
      return new Response(body, { headers: { 'Content-Length': String(files.get('SimpleSkin_geometry.bin')!.length) } });
    };
    await expect(loadGltf(rootUrl, { fetch: fetcher, maxBytes: files.get('SimpleSkin.gltf')!.length + 1 }))
      .rejects.toThrow(/GLTF_DOWNLOAD_BUDGET/);
    expect(requests).toHaveLength(2);
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('cancels a streaming reader when actual chunks exceed the cumulative budget', async () => {
    const cancel = vi.fn();
    const requests: URL[] = [];
    const binary = files.get('SimpleSkin_geometry.bin')!;
    let offset = 0;
    const fetcher: typeof fetch = async (input) => {
      const url = new URL(String(input));
      requests.push(url);
      if (url.pathname.endsWith('.gltf')) return response(files.get('SimpleSkin.gltf')!);
      const body = new ReadableStream<Uint8Array>({
        pull(controller) { controller.enqueue(binary.slice(offset, offset += 8)); },
        cancel,
      }, { highWaterMark: 0 });
      return new Response(body);
    };
    await expect(loadGltf(rootUrl, { fetch: fetcher, maxBytes: files.get('SimpleSkin.gltf')!.length + 12 }))
      .rejects.toThrow(/GLTF_DOWNLOAD_BUDGET/);
    expect(requests).toHaveLength(2);
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('passes AbortSignal to fetch and cancels an in-flight reader on abort', async () => {
    const controller = new AbortController();
    const cancel = vi.fn();
    const requests: URL[] = [];
    const binary = files.get('SimpleSkin_geometry.bin')!;
    const fetcher: typeof fetch = async (input, init) => {
      const url = new URL(String(input));
      requests.push(url);
      expect(init?.signal).toBe(controller.signal);
      if (url.pathname.endsWith('.gltf')) return response(files.get('SimpleSkin.gltf')!);
      let sent = false;
      return new Response(new ReadableStream<Uint8Array>({
        pull(stream) {
          if (!sent) { sent = true; stream.enqueue(binary.slice(0, 8)); controller.abort(); }
        },
        cancel,
      }, { highWaterMark: 0 }));
    };
    await expect(loadGltf(rootUrl, { fetch: fetcher, signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
    expect(requests).toHaveLength(2);
    expect(cancel).toHaveBeenCalledOnce();
    const alreadyAborted = new AbortController();
    alreadyAborted.abort();
    await expect(loadGltf(rootUrl, { fetch: fetcher, signal: alreadyAborted.signal })).rejects.toMatchObject({ name: 'AbortError' });
    expect(requests).toHaveLength(2);
  });

  it('reports HTTP errors, source buffer length mismatches, and disallowed protocols', async () => {
    await expect(loadGltf(rootUrl, { fetch: async () => new Response('missing', { status: 404 }) }))
      .rejects.toThrow(/GLTF_FETCH: 404/);
    const wrongLength = sourceJson();
    wrongLength.buffers[0].byteLength++;
    const modified = new TextEncoder().encode(JSON.stringify(wrongLength));
    await expect(loadGltf(rootUrl, { fetch: fixtureFetch([], modified) })).rejects.toThrow(/GLTF_BUFFER_LENGTH/);
    await expect(loadGltf('file:///scenes/SimpleSkin.gltf', { fetch: fixtureFetch([]) }))
      .rejects.toThrow(/GLTF_URL_PROTOCOL/);
    const wrongProtocol = sourceJson();
    wrongProtocol.buffers[0].uri = 'file:///private/SimpleSkin_geometry.bin';
    await expect(loadGltf(rootUrl, { fetch: fixtureFetch([], new TextEncoder().encode(JSON.stringify(wrongProtocol))) }))
      .rejects.toThrow(/GLTF_RESOURCE_URL_PROTOCOL/);
  });

  it('limits cumulative decoded image bytes independently of downloaded bytes', async () => {
    const json = sourceJson();
    json.images = [{ uri: 'tiny1.png' }, { uri: 'tiny2.png' }];
    json.textures = [{ source: 0 }, { source: 1 }];
    const modified = new TextEncoder().encode(JSON.stringify(json));
    const close = vi.fn(), canvas = vi.fn();
    vi.stubGlobal('createImageBitmap', vi.fn().mockImplementation(async () => ({ width: 1, height: 1, close })));
    vi.stubGlobal('OffscreenCanvas', class {
      constructor(width: number, height: number) { canvas(width, height); }
      getContext() { return { drawImage: () => {}, getImageData: () => ({ data: new Uint8ClampedArray([1, 2, 3, 4]) }) }; }
    });
    const requests: URL[] = [];
    const baseFetch = fixtureFetch(requests, modified);
    const fetcher: typeof fetch = async (input, init) => {
      const url = new URL(String(input));
      if (url.pathname.endsWith('.png')) { requests.push(url); return response(new Uint8Array([1])); }
      return baseFetch(input, init);
    };
    const asset = await loadGltf(rootUrl, { fetch: fetcher, maxDecodedImageBytes: 8 });
    expect(asset.skins[0].joints).toEqual([1, 2]);
    expect(canvas).toHaveBeenCalledTimes(2);
    expect(close).toHaveBeenCalledTimes(2);
    canvas.mockClear(); close.mockClear(); requests.length = 0;
    await expect(loadGltf(rootUrl, { fetch: fetcher, maxDecodedImageBytes: 7 })).rejects.toThrow(/GLTF_DECODED_IMAGE_BUDGET/);
    expect(canvas).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(requests.some(url => url.pathname.endsWith('tiny2.png'))).toBe(false);
    await expect(loadGltf(rootUrl, { fetch: fetcher, maxDecodedImageBytes: -1 })).rejects.toThrow(/GLTF_DECODED_IMAGE_BUDGET/);
  });

  it('skips fetching and decoding an unreferenced image', async () => {
    const json = sourceJson();
    json.images = [{ uri: 'unused.png' }];
    const requests: URL[] = [];
    const decode = vi.fn();
    vi.stubGlobal('createImageBitmap', decode);
    const asset = await loadGltf(rootUrl, { fetch: fixtureFetch(requests, new TextEncoder().encode(JSON.stringify(json))), maxDecodedImageBytes: 0 });
    expect(asset.skins[0].joints).toEqual([1, 2]);
    expect(requests.some(url => url.pathname.endsWith('unused.png'))).toBe(false);
    expect(decode).not.toHaveBeenCalled();
  });

  it('rejects malformed image sources and out-of-declaration buffer views before decoding', async () => {
    const decode = vi.fn();
    vi.stubGlobal('createImageBitmap', decode);
    const invalid = (mutate: (json: FixtureJson) => void) => {
      const json = sourceJson();
      mutate(json);
      return loadGltf(rootUrl, { fetch: fixtureFetch([], new TextEncoder().encode(JSON.stringify(json))) });
    };
    await expect(invalid(json => { json.images = [{}]; })).rejects.toThrow(/GLTF_IMAGE_SOURCE/);
    await expect(invalid(json => { json.images = [{ uri: 'tiny.png', bufferView: 0, mimeType: 'image/png' }]; }))
      .rejects.toThrow(/GLTF_IMAGE_SOURCE/);
    await expect(invalid(json => { json.images = [{ bufferView: 0 }]; }))
      .rejects.toThrow(/GLTF_IMAGE_FORMAT/);
    await expect(invalid(json => { json.images = [{ bufferView: 0.5, mimeType: 'image/png' }]; }))
      .rejects.toThrow(/GLTF_IMAGE_BUFFER_VIEW/);
    await expect(invalid(json => {
      json.bufferViews.push({ buffer: -1, byteOffset: 0, byteLength: 4 });
      json.images = [{ bufferView: json.bufferViews.length - 1, mimeType: 'image/png' }];
    })).rejects.toThrow(/GLTF_IMAGE_BUFFER_VIEW/);
    await expect(invalid(json => {
      json.buffers[0].byteLength = 160; // Download still has 168 bytes.
      json.bufferViews.push({ buffer: 0, byteOffset: 158, byteLength: 4 });
      json.images = [{ bufferView: json.bufferViews.length - 1, mimeType: 'image/png' }];
    })).rejects.toThrow(/GLTF_IMAGE_BOUNDS/);
    expect(decode).not.toHaveBeenCalled();
  });
});
