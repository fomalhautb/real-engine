import { afterEach, describe, expect, it, vi } from 'vitest';
import { clampMaterialFactors, loadAssetSetRoots, loadGltfPackageScene } from '../src/gltf-package-scene';

const roots = { geometryRoot: '/artifacts/g/', textureGpuRoot: '/artifacts/t/', textureReferenceRoot: '/artifacts/t/', uvAuditUrl: '/artifacts/uv.json' };
const serve = (doc: unknown) => vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(doc))));

describe('glTF package scene', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('clamps only float32 rounding excess in material factors and reports each edit', () => {
    const doc = { materials: [{ pbrMetallicRoughness: { baseColorFactor: [1.0000001, 0.5, 1, 1], roughnessFactor: 1 + 2 ** -23 }, emissiveFactor: [0, 0, 0] }] };
    expect(clampMaterialFactors(doc).map(edit => edit.path)).toEqual([
      'materials[0].pbrMetallicRoughness.baseColorFactor[0]', 'materials[0].pbrMetallicRoughness.roughnessFactor']);
    expect(doc.materials[0].pbrMetallicRoughness).toEqual({ baseColorFactor: [1, 0.5, 1, 1], roughnessFactor: 1 });
    expect(() => clampMaterialFactors({ materials: [{ emissiveFactor: [1.5, 0, 0] }] })).toThrow('GLTF_PACKAGE_FACTOR_RANGE: materials[0].emissiveFactor[0]');
  });

  it('reads local asset-set roots and rejects remote, traversing or foreign documents', async () => {
    serve({ format: 'real-engine-asset-set', version: 1, roots: { ...roots, cameraPathUrl: '/artifacts/set/camera-path.json' } });
    await expect(loadAssetSetRoots('/artifacts/set/asset-set.json')).resolves.toEqual({ ...roots, cameraPathUrl: '/artifacts/set/camera-path.json' });
    await expect(loadAssetSetRoots('https://example.com/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ROOT: assetSet');
    serve({ format: 'real-engine-asset-set', version: 1, roots: { ...roots, geometryRoot: '/artifacts/../g/' } });
    await expect(loadAssetSetRoots('/artifacts/set/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ROOT: geometryRoot');
    serve({ format: 'other', version: 1, roots });
    await expect(loadAssetSetRoots('/artifacts/set/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ASSET_SET_FORMAT');
  });

  it('returns the fine query root and recorded manifest hashes, and enforces them', async () => {
    const hash = 'ab'.repeat(32);
    serve({ format: 'real-engine-asset-set', version: 1, roots: { ...roots, fineQueryRoot: '/artifacts/q/' },
      packages: { geometry: { manifestSha256: hash }, fineQuery: { manifestSha256: hash } } });
    const loaded = await loadAssetSetRoots('/artifacts/set/asset-set.json');
    expect(loaded).toEqual({ ...roots, fineQueryRoot: '/artifacts/q/', manifestSha256: { geometry: hash, fineQuery: hash } });
    serve({ format: 'real-engine-asset-set', version: 1, roots: { ...roots, fineQueryRoot: 'https://example.com/q/' } });
    await expect(loadAssetSetRoots('/artifacts/set/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ROOT: fineQueryRoot');
    serve({ format: 'real-engine-asset-set', version: 1, roots, packages: { textures: { manifestSha256: 'not-a-hash' } } });
    await expect(loadAssetSetRoots('/artifacts/set/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ASSET_SET_HASH: textures');
    // A geometry manifest whose bytes do not hash to the recorded value is rejected before anything else is read.
    serve({ version: 1, format: 'real-engine-gltf-geometry', primitives: [] });
    await expect(loadGltfPackageScene(loaded)).rejects.toThrow('GLTF_PACKAGE_MANIFEST_HASH: /artifacts/g/manifest.json');
  });
});
