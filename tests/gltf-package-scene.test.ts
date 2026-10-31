import { afterEach, describe, expect, it, vi } from 'vitest';
import { clampMaterialFactors, loadAssetSetRoots } from '../src/gltf-package-scene';

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
});
