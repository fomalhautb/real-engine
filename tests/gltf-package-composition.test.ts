import { createHash } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { StreamedSceneBatch } from '../src/contracts';
import { loadAssetComposition, loadComposedPackageScene, type AssetComposition } from '../src/gltf-package-composition';
import { loadGltfPackageScene, type GltfPackageScene } from '../src/gltf-package-scene';
import type { PreparedTextureAsset } from '../src/texture-storage';
import type { TextureStreamMip, TextureStreamSource } from '../src/texture-stream-types';

vi.mock('../src/gltf-package-scene', async importOriginal => ({
  ...await importOriginal<typeof import('../src/gltf-package-scene')>(), loadGltfPackageScene: vi.fn() }));
const loadScene = vi.mocked(loadGltfPackageScene);

const sha256 = (text: string | Uint8Array): string => createHash('sha256').update(text).digest('hex');
type Json = Record<string, any>;

const roots = (name: string, camera = false): Json => ({ geometryRoot: `/artifacts/${name}-geometry/`, textureGpuRoot: `/artifacts/${name}-textures/`,
  textureReferenceRoot: `/artifacts/${name}-textures/`, uvAuditUrl: `/artifacts/${name}-uv.json`,
  ...(camera ? { cameraPathUrl: `/artifacts/${name}-set/camera-path.json` } : {}) });
const set = (name: string, camera = false): Json => ({ version: 1, format: 'real-engine-asset-set', name: 'test-' + name,
  source: { file: `source/${name}.gltf`, bytes: 10, sha256: sha256('source-' + name) }, roots: roots(name, camera), packages: {}, license: null, omitted: [] });

/** Serves `files` (url -> text) through a stubbed fetch; anything else is a 404. */
function serve(files: Map<string, string>): void {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => files.has(url) ? new Response(files.get(url)) : new Response('missing', { status: 404 })));
}
function published(sets: Json[], edit: (composition: Json) => void = () => {}): Map<string, string> {
  const files = new Map<string, string>();
  const components = sets.map(doc => {
    const name = doc.name.slice('test-'.length), text = JSON.stringify(doc, null, 2) + '\n', url = `/artifacts/${name}-set/asset-set.json`;
    files.set(url, text);
    return { name, setName: doc.name, assetSet: { url, bytes: new TextEncoder().encode(text).length, sha256: sha256(text) }, source: doc.source };
  });
  const composition: Json = { version: 1, format: 'real-engine-asset-composition', name: 'test-composed', frame: { componentTransform: 'identity', note: 'test' },
    components, cameraPath: { component: 'main', url: '/artifacts/main-set/camera-path.json', sha256: sha256('camera'), samples: 2, durationSeconds: 1 },
    license: null, totals: {} };
  edit(composition);
  files.set('/artifacts/composed/asset-set.json', JSON.stringify(composition));
  return files;
}

function textureSource(id: string, fill: number): TextureStreamSource {
  const mips: TextureStreamMip[] = [], levels: PreparedTextureAsset['levels'][number][] = [];
  for (const [level, size] of [[0, 4], [1, 2], [2, 1]]) {
    const data = new Uint8Array(size * size * 4).fill(fill + level);
    mips.push({ level, width: size, height: size, byteLength: data.length, sha256: sha256(data) });
    if (level >= 1) levels.push({ width: size, height: size, data });
  }
  return { id, width: 4, height: 4, format: 'rgba8', colorSpace: 'linear', mips,
    fallback: { kind: 'prepared', width: 4, height: 4, format: 'rgba8', colorSpace: 'linear', baseMipLevel: 1, levels },
    provider: async mip => new Uint8Array(mip.byteLength) };
}
const noMask = (): GltfPackageScene['mask'] => ({ primitives: 0, sourceVertices: 0, sourceTriangles: 0, triangles: 0, meshBytes: 0, sourceBufferBytes: 0, budget: null });
/** A package scene as loadGltfPackageScene returns it: instances numbered from 1 per scene. */
function packageScene(name: string, instancesPerBatch: number[], offset: number): GltfPackageScene {
  let entityId = 1;
  const batches = instancesPerBatch.map((count, i) => ({ name: `x:${i}:0`, capacity: count,
    instances: Array.from({ length: count }, () => ({ entityId: entityId++, model: [], normalMatrix: [], bones: [], deformScalars: [],
      color: [1, 1, 1, 1], metalness: 1, roughness: 1, historyVersion: 0 })) }) as unknown as StreamedSceneBatch);
  return { batches, sources: [textureSource('image-0', offset), textureSource('image-1', offset + 3)],
    bounds: { min: [offset, -1, 0], max: [offset + 1, 2, 3] },
    omitted: [{ mesh: 1, primitive: 0, material: 2, route: 'source', coverage: 'blend', nodes: [4], sourceTriangles: 12, reason: 'blend' }],
    conversions: [{ path: 'materials[0].pbrMetallicRoughness.roughnessFactor', before: 1.0000001, after: 1 }],
    unsupportedMaterialInputs: [{ path: 'materials[1].occlusionTexture', reason: 'not read' }],
    sourceInstances: 10 * offset, pagedTriangles: 1000 * offset, omittedTriangles: 12, mask: noMask(),
    queryProxies: { root: `/artifacts/${name}-geometry/`, fine: false, maxError: null, primitives: 4, triangles: 100 * offset },
    assetIdentity: sha256('identity-' + name) };
}
/** Two components named main and ivy, without fetching their sets. */
const twoComponents = (): AssetComposition => ({ name: 'test-composed', cameraPath: null, components: ['main', 'ivy'].map(name => ({ name, setName: 'test-' + name,
  assetSet: { url: `/artifacts/${name}-set/asset-set.json`, bytes: 1, sha256: sha256(name) }, sourceSha256: sha256('source-' + name),
  roots: roots(name) as AssetComposition['components'][number]['roots'] })) });

describe('glTF package composition', () => {
  beforeEach(() => { loadScene.mockReset(); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('reads a composition and verifies every component set by length and SHA-256', async () => {
    serve(published([set('main', true), set('ivy')]));
    const composition = await loadAssetComposition('/artifacts/composed/asset-set.json');
    expect(composition).toEqual({ name: 'test-composed', cameraPath: { component: 'main', url: '/artifacts/main-set/camera-path.json', sha256: sha256('camera') },
      components: [
        { name: 'main', setName: 'test-main', assetSet: expect.objectContaining({ url: '/artifacts/main-set/asset-set.json' }),
          sourceSha256: sha256('source-main'), roots: roots('main', true) },
        { name: 'ivy', setName: 'test-ivy', assetSet: expect.objectContaining({ url: '/artifacts/ivy-set/asset-set.json' }),
          sourceSha256: sha256('source-ivy'), roots: roots('ivy') }] });

    // A component set that changed after composition (same length, different bytes / different length).
    const files = published([set('main', true), set('ivy')]);
    files.set('/artifacts/ivy-set/asset-set.json', files.get('/artifacts/ivy-set/asset-set.json')!.replace('test-ivy', 'test-ivz'));
    serve(files);
    await expect(loadAssetComposition('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_HASH: /artifacts/ivy-set/asset-set.json');
    files.set('/artifacts/ivy-set/asset-set.json', files.get('/artifacts/ivy-set/asset-set.json')! + ' ');
    await expect(loadAssetComposition('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_BYTE_BUDGET');
    files.set('/artifacts/ivy-set/asset-set.json', '{}');
    await expect(loadAssetComposition('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_BYTE_LENGTH');
    files.delete('/artifacts/ivy-set/asset-set.json');
    await expect(loadAssetComposition('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_HTTP: /artifacts/ivy-set/asset-set.json');
  });

  it('rejects foreign documents, remote or traversing paths, duplicates and inconsistent camera paths', async () => {
    const url = '/artifacts/composed/asset-set.json';
    await expect(loadAssetComposition('https://example.com/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_PATH: composition');
    serve(new Map([[url, JSON.stringify(set('main'))]]));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_FORMAT');
    serve(published([set('main', true)], c => { c.frame.componentTransform = 'translate'; }));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_FRAME');
    serve(published([set('main', true), set('ivy')], c => { c.components[1].name = 'main'; }));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_COMPONENT_NAME: main');
    serve(published([set('main', true), set('ivy')], c => { c.components[1].assetSet.url = '//cdn.example.com/asset-set.json'; }));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_PATH: ivy assetSet');
    serve(published([set('main', true), set('ivy')], c => { c.components[1].source = { sha256: sha256('source-main') }; }));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_SOURCE: ivy');
    const twin = set('ivy');
    twin.source = set('main').source;
    serve(published([set('main', true), twin]));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_DUPLICATE_SOURCE: ivy');
    const traversing = set('ivy');
    traversing.roots.geometryRoot = '/artifacts/../ivy-geometry/';
    serve(published([set('main', true), traversing]));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_PATH: ivy geometryRoot');
    serve(published([set('main', true), set('ivy')], c => { c.cameraPath.component = 'ivy'; }));
    await expect(loadAssetComposition(url)).rejects.toThrow('GLTF_COMPOSITION_CAMERA_PATH');
    serve(published([set('main'), set('ivy')], c => { c.cameraPath = null; }));
    await expect(loadAssetComposition(url)).resolves.toMatchObject({ cameraPath: null });
  });

  it('loads components in order and merges them with prefixed names and ids and renumbered entities', async () => {
    serve(published([set('main', true), set('ivy')]));
    const composition = await loadAssetComposition('/artifacts/composed/asset-set.json');
    const main = packageScene('main', [2, 1], 1), ivy = packageScene('ivy', [3], 5);
    loadScene.mockResolvedValueOnce(main).mockResolvedValueOnce(ivy);
    const events: [string, unknown][] = [], signal = new AbortController().signal;
    const scene = await loadComposedPackageScene(composition, { anisotropy: 8, signal, progress: (phase, data) => events.push([phase, data]) });

    expect(loadScene.mock.calls.map(([r, o]) => [r, o?.name, o?.maxBatches, o?.anisotropy, o?.signal])).toEqual([
      [roots('main', true), 'test-composed/main', 4096, 8, signal], [roots('ivy'), 'test-composed/ivy', 4094, 8, signal]]);
    loadScene.mock.calls[1][1]?.progress?.('textures-ready', { count: 2 });
    expect(events).toEqual([['component-ready', { component: 'main', data: { batches: 2, textures: 2 } }],
      ['component-ready', { component: 'ivy', data: { batches: 1, textures: 2 } }], ['textures-ready', { component: 'ivy', data: { count: 2 } }]]);

    // Entity ids continue across components; the loaded package scenes are not mutated.
    expect(scene.batches.map(b => b.instances.map(i => i.entityId))).toEqual([[1, 2], [3], [4, 5, 6]]);
    expect(ivy.batches[0].instances.map(i => i.entityId)).toEqual([1, 2, 3]);
    expect(scene.batches[2].instances[0]).toEqual({ ...ivy.batches[0].instances[0], entityId: 4 });
    expect(scene.batches[2].capacity).toBe(3);
    // Texture source ids become unique; fallback identity (what materials reference) is kept.
    expect(scene.sources.map(s => s.id)).toEqual(['main/image-0', 'main/image-1', 'ivy/image-0', 'ivy/image-1']);
    expect(scene.sources[2].fallback).toBe(ivy.sources[0].fallback);
    expect(scene.sources[2].provider).toBe(ivy.sources[0].provider);
    expect(new Set(scene.sources.map(s => s.fallback)).size).toBe(4);
    expect(Object.isFrozen(scene.sources[0])).toBe(true);

    expect(scene.bounds).toEqual({ min: [1, -1, 0], max: [6, 2, 3] });
    expect(scene.omitted.map(o => [o.component, o.mesh, o.sourceTriangles])).toEqual([['main', 1, 12], ['ivy', 1, 12]]);
    expect(scene.conversions.map(c => c.component)).toEqual(['main', 'ivy']);
    expect(scene.unsupportedMaterialInputs).toEqual([{ component: 'main', path: 'materials[1].occlusionTexture', reason: 'not read' },
      { component: 'ivy', path: 'materials[1].occlusionTexture', reason: 'not read' }]);
    expect(scene).toMatchObject({ sourceInstances: 60, pagedTriangles: 6000, omittedTriangles: 24 });
    expect(scene.components).toEqual([
      { name: 'main', assetIdentity: sha256('identity-main'), batches: 2, sources: 2, firstEntityId: 1, entities: 3, sourceInstances: 10, pagedTriangles: 1000, omittedTriangles: 12,
        mask: main.mask, queryProxies: main.queryProxies },
      { name: 'ivy', assetIdentity: sha256('identity-ivy'), batches: 1, sources: 2, firstEntityId: 4, entities: 3, sourceInstances: 50, pagedTriangles: 5000, omittedTriangles: 12,
        mask: ivy.mask, queryProxies: ivy.queryProxies }]);
    expect(scene.mask).toEqual(noMask());
  });

  it('passes each component set\'s fine query root and manifest hashes to the package loader', async () => {
    const ivy = set('ivy');
    ivy.roots.fineQueryRoot = '/artifacts/ivy-query/';
    ivy.packages = { geometry: { manifestSha256: sha256('geometry') }, fineQuery: { manifestSha256: sha256('query') } };
    serve(published([set('main', true), ivy]));
    const composition = await loadAssetComposition('/artifacts/composed/asset-set.json');
    expect(composition.components.map(c => c.roots)).toEqual([roots('main', true), { ...roots('ivy'), fineQueryRoot: '/artifacts/ivy-query/',
      manifestSha256: { geometry: sha256('geometry'), fineQuery: sha256('query') } }]);
    ivy.packages.textures = { manifestSha256: 'not-a-hash' };
    serve(published([set('main', true), ivy]));
    await expect(loadAssetComposition('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_COMPOSITION_COMPONENT_HASH: ivy textures');
  });

  it('counts the MASK geometry of all components against one budget', async () => {
    const budget = { maxPrimitives: 10, maxVertices: 1000, maxTriangles: 500 }, left = { maxPrimitives: 7, maxVertices: 600, maxTriangles: 300 };
    const main = { ...packageScene('main', [1], 1), mask: { primitives: 3, sourceVertices: 400, sourceTriangles: 200, triangles: 600, meshBytes: 9000, sourceBufferBytes: 7000, budget } };
    const ivy = { ...packageScene('ivy', [1], 2), mask: { primitives: 1, sourceVertices: 100, sourceTriangles: 50, triangles: 50, meshBytes: 1000, sourceBufferBytes: 7000, budget: left } };
    loadScene.mockResolvedValueOnce(main).mockResolvedValueOnce(ivy);
    const scene = await loadComposedPackageScene(twoComponents(), { mask: budget });
    // The second component may only use what the first one left.
    expect(loadScene.mock.calls.map(([, options]) => options?.mask)).toEqual([budget, left]);
    expect(scene.mask).toEqual({ primitives: 4, sourceVertices: 500, sourceTriangles: 250, triangles: 650, meshBytes: 10000, sourceBufferBytes: 14000, budget });
    expect(scene.components.map(c => c.mask.primitives)).toEqual([3, 1]);
    loadScene.mockReset();
    loadScene.mockResolvedValueOnce(packageScene('main', [1], 1)).mockResolvedValueOnce(packageScene('ivy', [1], 2));
    await loadComposedPackageScene(twoComponents(), { mask: false });
    expect(loadScene.mock.calls.map(([, options]) => options?.mask)).toEqual([false, false]);
  });

  it('keeps the batch budget for the whole composition and rejects repeated assets or aborted loads', async () => {
    const composition = twoComponents();
    loadScene.mockResolvedValueOnce(packageScene('main', [1, 1], 1));
    await expect(loadComposedPackageScene(composition, { maxBatches: 2 })).rejects.toThrow('GLTF_COMPOSITION_BATCH_BUDGET');
    expect(loadScene).toHaveBeenCalledTimes(1);
    loadScene.mockReset();
    loadScene.mockResolvedValueOnce(packageScene('main', [1], 1)).mockResolvedValueOnce(packageScene('main', [1], 2));
    await expect(loadComposedPackageScene(composition)).rejects.toThrow('GLTF_COMPOSITION_DUPLICATE_ASSET: ivy');
    const controller = new AbortController();
    controller.abort();
    loadScene.mockReset();
    await expect(loadComposedPackageScene(composition, { signal: controller.signal })).rejects.toThrow();
    expect(loadScene).not.toHaveBeenCalled();
  });
});
