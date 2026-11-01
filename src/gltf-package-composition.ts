/**
 * Loads an asset composition (format real-engine-asset-composition, written by
 * `scripts/write-asset-set.mjs --compose`): finished single-source asset sets
 * whose sources share one glTF scene space, such as the Intel Sponza building
 * and its add-ons. Every component set is fetched bounded and SHA-256 checked
 * against the composition, then loaded with loadGltfPackageScene. The merged
 * scene prefixes batch names and texture source ids with the component name,
 * renumbers entity ids so they stay unique across components, and tags
 * omitted primitives, conversions and unsupported inputs with their component.
 * Full-detail MASK geometry of all components counts against one MASK budget.
 * Components get no extra transform. Loading creates no GPU resources.
 */
import type { SceneBatch, StreamedSceneBatch } from './contracts';
import {
  loadGltfPackageScene, type GltfPackageMaskBudget, type GltfPackageMaskSummary, type GltfPackageRoots, type GltfPackageScene,
  type GltfPackageSceneOptions, type OmittedPrimitive,
} from './gltf-package-scene';
import { validateTextureStreamSource } from './texture-stream-source';
import type { TextureStreamSource } from './texture-stream-types';

const MiB = 1024 ** 2;
export const ASSET_COMPOSITION_LIMITS = Object.freeze({ compositionBytes: MiB, setBytes: MiB, components: 64, maxEntityId: 0xffffffff });
const COMPONENT_NAME = /^[a-z][a-z0-9-]{0,31}$/;
const COMPOSITION_NAME = /^[a-z0-9][a-z0-9.-]{0,63}$/;
const SHA256 = /^[0-9a-f]{64}$/;

export interface AssetCompositionComponent {
  name: string;
  /** Name recorded inside the component's own asset set. */
  setName: string;
  assetSet: { url: string; bytes: number; sha256: string };
  sourceSha256: string;
  roots: GltfPackageRoots;
}

export interface AssetComposition {
  name: string;
  components: AssetCompositionComponent[];
  /** Camera path chosen when the composition was written (the file of one component set). */
  cameraPath: { component: string; url: string; sha256: string } | null;
}

export interface ComposedOmittedPrimitive extends OmittedPrimitive { component: string }

export interface ComposedPackageSceneOptions extends Omit<GltfPackageSceneOptions, 'name' | 'maxBatches' | 'progress'> {
  /** Batch name prefix (default: the composition name); batches are named `<prefix>/<component>:<mesh>:<primitive>`. */
  name?: string;
  /** Batch budget of the whole composed scene (default 4096). */
  maxBatches?: number;
  progress?: (phase: string, data: { component: string; data?: unknown }) => void;
}

export interface ComposedPackageScene {
  /** Streamed (paged) batches and full-detail MASK batches of all components, in composition order. */
  batches: (StreamedSceneBatch | SceneBatch)[];
  /** Texture sources with ids `<component>/<source id>`; fallback identities are unchanged. */
  sources: TextureStreamSource[];
  bounds: { min: [number, number, number]; max: [number, number, number] };
  omitted: ComposedOmittedPrimitive[];
  conversions: { component: string; path: string; before: number; after: number }[];
  unsupportedMaterialInputs: { component: string; path: string; reason: string }[];
  components: { name: string; assetIdentity: string; batches: number; sources: number; firstEntityId: number; entities: number;
    sourceInstances: number; pagedTriangles: number; omittedTriangles: number; mask: GltfPackageMaskSummary;
    queryProxies: GltfPackageScene['queryProxies'] }[];
  sourceInstances: number;
  /** Instance-expanded triangles of the streamed (paged) batches of all components. */
  pagedTriangles: number;
  /** Instance-expanded triangles of omitted primitives of all components. */
  omittedTriangles: number;
  /** MASK geometry of all components; `budget` is the one budget they share (null when MASK primitives are omitted). */
  mask: GltfPackageMaskSummary;
}

function localPath(value: unknown, label: string, directory: boolean): string {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || value.includes('..') || value.includes('\\')
    || value.endsWith('/') !== directory) throw new Error('GLTF_COMPOSITION_PATH: ' + label);
  return value;
}

/** Reads at most `limit` bytes; with `expected`, requires exactly that length and SHA-256. */
async function fetchBounded(url: string, limit: number, signal: AbortSignal | undefined,
  expected?: { bytes: number; sha256: string }): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(url, { signal });
  if (!response.ok || !response.body) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error('GLTF_COMPOSITION_HTTP: ' + url);
  }
  const reader = response.body.getReader(), data = new Uint8Array(limit);
  let offset = 0;
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      if (next.value.length > limit - offset) throw new Error('GLTF_COMPOSITION_BYTE_BUDGET: ' + url);
      data.set(next.value, offset); offset += next.value.length;
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; } finally { reader.releaseLock(); }
  const bytes = data.slice(0, offset);
  if (expected) {
    if (offset !== expected.bytes) throw new Error('GLTF_COMPOSITION_BYTE_LENGTH: ' + url);
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    if (Array.from(digest, n => n.toString(16).padStart(2, '0')).join('') !== expected.sha256) throw new Error('GLTF_COMPOSITION_HASH: ' + url);
  }
  return bytes;
}

function parse(bytes: Uint8Array, url: string): any {
  let value: unknown;
  try { value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); } catch { throw new Error('GLTF_COMPOSITION_JSON: ' + url); }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new Error('GLTF_COMPOSITION_JSON: ' + url);
  return value;
}

/** Reads a composition and every component asset set it names, verifying each set's length and SHA-256. */
export async function loadAssetComposition(url: string, signal?: AbortSignal): Promise<AssetComposition> {
  const doc = parse(await fetchBounded(localPath(url, 'composition', false), ASSET_COMPOSITION_LIMITS.compositionBytes, signal), url);
  if (doc.format !== 'real-engine-asset-composition' || doc.version !== 1 || !Array.isArray(doc.components)) throw new Error('GLTF_COMPOSITION_FORMAT');
  if (typeof doc.name !== 'string' || !COMPOSITION_NAME.test(doc.name)) throw new Error('GLTF_COMPOSITION_NAME');
  // Version 1 places every component with its own glTF transforms only.
  if (doc.frame?.componentTransform !== 'identity') throw new Error('GLTF_COMPOSITION_FRAME');
  if (doc.components.length < 1 || doc.components.length > ASSET_COMPOSITION_LIMITS.components) throw new Error('GLTF_COMPOSITION_COMPONENT_COUNT');
  const components: AssetCompositionComponent[] = [], sources = new Set<string>();
  for (const entry of doc.components) {
    const name = entry?.name;
    if (typeof name !== 'string' || !COMPONENT_NAME.test(name) || components.some(c => c.name === name)) {
      throw new Error('GLTF_COMPOSITION_COMPONENT_NAME: ' + String(name));
    }
    const record = entry.assetSet ?? {};
    const setUrl = localPath(record.url, name + ' assetSet', false);
    if (!Number.isSafeInteger(record.bytes) || record.bytes < 1 || record.bytes > ASSET_COMPOSITION_LIMITS.setBytes
      || typeof record.sha256 !== 'string' || !SHA256.test(record.sha256)) throw new Error('GLTF_COMPOSITION_COMPONENT_SET: ' + name);
    const set = parse(await fetchBounded(setUrl, record.bytes, signal, { bytes: record.bytes, sha256: record.sha256 }), setUrl);
    if (set.format !== 'real-engine-asset-set' || set.version !== 1 || !set.roots) throw new Error('GLTF_COMPOSITION_COMPONENT_FORMAT: ' + name);
    const sourceSha256 = set.source?.sha256;
    if (typeof sourceSha256 !== 'string' || !SHA256.test(sourceSha256) || sourceSha256 !== entry.source?.sha256) throw new Error('GLTF_COMPOSITION_SOURCE: ' + name);
    if (sources.has(sourceSha256)) throw new Error('GLTF_COMPOSITION_DUPLICATE_SOURCE: ' + name);
    sources.add(sourceSha256);
    const roots: GltfPackageRoots = {
      geometryRoot: localPath(set.roots.geometryRoot, name + ' geometryRoot', true),
      textureGpuRoot: localPath(set.roots.textureGpuRoot, name + ' textureGpuRoot', true),
      textureReferenceRoot: localPath(set.roots.textureReferenceRoot, name + ' textureReferenceRoot', true),
      uvAuditUrl: localPath(set.roots.uvAuditUrl, name + ' uvAuditUrl', false),
    };
    if (set.roots.cameraPathUrl !== undefined) roots.cameraPathUrl = localPath(set.roots.cameraPathUrl, name + ' cameraPathUrl', false);
    if (set.roots.fineQueryRoot !== undefined) roots.fineQueryRoot = localPath(set.roots.fineQueryRoot, name + ' fineQueryRoot', true);
    // Manifest hashes recorded by the component set; loadGltfPackageScene checks each fetched manifest against them.
    const hashes = Object.entries({ geometry: set.packages?.geometry?.manifestSha256, textures: set.packages?.textures?.manifestSha256,
      fineQuery: set.packages?.fineQuery?.manifestSha256 }).filter(([, value]) => value !== undefined);
    for (const [kind, value] of hashes) if (typeof value !== 'string' || !SHA256.test(value)) throw new Error(`GLTF_COMPOSITION_COMPONENT_HASH: ${name} ${kind}`);
    if (hashes.length) roots.manifestSha256 = Object.fromEntries(hashes);
    components.push({ name, setName: String(set.name), assetSet: { url: setUrl, bytes: record.bytes, sha256: record.sha256 }, sourceSha256, roots });
  }
  let cameraPath: AssetComposition['cameraPath'] = null;
  if (doc.cameraPath !== null && doc.cameraPath !== undefined) {
    const owner = components.find(c => c.name === doc.cameraPath.component);
    if (!owner || owner.roots.cameraPathUrl === undefined || doc.cameraPath.url !== owner.roots.cameraPathUrl
      || typeof doc.cameraPath.sha256 !== 'string' || !SHA256.test(doc.cameraPath.sha256)) throw new Error('GLTF_COMPOSITION_CAMERA_PATH');
    cameraPath = { component: owner.name, url: owner.roots.cameraPathUrl, sha256: doc.cameraPath.sha256 };
  }
  return { name: doc.name, components, cameraPath };
}

/** Loads every component with loadGltfPackageScene, in composition order, and merges the scenes. */
export async function loadComposedPackageScene(composition: AssetComposition, options: ComposedPackageSceneOptions = {}): Promise<ComposedPackageScene> {
  const { signal, progress = () => {} } = options, prefix = options.name ?? composition.name;
  const maxBatches = options.maxBatches ?? 4096;
  if (!Number.isSafeInteger(maxBatches) || maxBatches < 1) throw new Error('GLTF_COMPOSITION_BATCH_BUDGET');
  if (!composition.components.length || composition.components.length > ASSET_COMPOSITION_LIMITS.components) throw new Error('GLTF_COMPOSITION_COMPONENT_COUNT');
  const result: ComposedPackageScene = { batches: [], sources: [], bounds: { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] },
    omitted: [], conversions: [], unsupportedMaterialInputs: [], components: [], sourceInstances: 0, pagedTriangles: 0, omittedTriangles: 0,
    mask: { primitives: 0, sourceVertices: 0, sourceTriangles: 0, triangles: 0, meshBytes: 0, sourceBufferBytes: 0, budget: null } };
  // One MASK budget covers the composition: each component may use what the earlier components left.
  let mask: GltfPackageMaskBudget | false | undefined = options.mask;
  const names = new Set<string>(), identities = new Set<string>();
  let nextEntityId = 1;
  for (const component of composition.components) {
    signal?.throwIfAborted();
    if (!COMPONENT_NAME.test(component.name) || names.has(component.name)) throw new Error('GLTF_COMPOSITION_COMPONENT_NAME: ' + component.name);
    names.add(component.name);
    const remaining = maxBatches - result.batches.length;
    if (remaining < 1) throw new Error('GLTF_COMPOSITION_BATCH_BUDGET');
    const scene = await loadGltfPackageScene(component.roots, { anisotropy: options.anisotropy, signal, name: `${prefix}/${component.name}`,
      maxBatches: remaining, mask, progress: (phase, data) => progress(phase, { component: component.name, data }) });
    if (identities.has(scene.assetIdentity)) throw new Error('GLTF_COMPOSITION_DUPLICATE_ASSET: ' + component.name);
    identities.add(scene.assetIdentity);
    // Each component numbers its instances from 1; shift them past the previous components.
    const offset = nextEntityId - 1, firstEntityId = nextEntityId;
    let last = offset;
    for (const batch of scene.batches) {
      const instances = batch.instances.map(instance => {
        if (!Number.isSafeInteger(instance.entityId) || instance.entityId < 1) throw new Error('GLTF_COMPOSITION_ENTITY_ID: ' + batch.name);
        const entityId = instance.entityId + offset;
        if (entityId > ASSET_COMPOSITION_LIMITS.maxEntityId) throw new Error('GLTF_COMPOSITION_ENTITY_BUDGET');
        last = Math.max(last, entityId);
        return { ...instance, entityId };
      });
      result.batches.push({ ...batch, instances });
    }
    nextEntityId = last + 1;
    // Package-local ids (image-0, image-1, ...) repeat across components; the streamer requires unique ids.
    for (const source of scene.sources) result.sources.push(validateTextureStreamSource({ ...source, id: `${component.name}/${source.id}` }));
    for (let axis = 0; axis < 3; axis++) {
      result.bounds.min[axis] = Math.min(result.bounds.min[axis], scene.bounds.min[axis]);
      result.bounds.max[axis] = Math.max(result.bounds.max[axis], scene.bounds.max[axis]);
    }
    for (const entry of scene.omitted) result.omitted.push({ component: component.name, ...entry });
    for (const entry of scene.conversions) result.conversions.push({ component: component.name, ...entry });
    for (const entry of scene.unsupportedMaterialInputs) result.unsupportedMaterialInputs.push({ component: component.name, ...entry });
    const used = scene.mask;
    if (used.budget) {
      result.mask.budget ??= used.budget;
      mask = { maxPrimitives: used.budget.maxPrimitives - used.primitives, maxVertices: used.budget.maxVertices - used.sourceVertices,
        maxTriangles: used.budget.maxTriangles - used.sourceTriangles };
    }
    for (const key of ['primitives', 'sourceVertices', 'sourceTriangles', 'triangles', 'meshBytes', 'sourceBufferBytes'] as const) result.mask[key] += used[key];
    result.components.push({ name: component.name, assetIdentity: scene.assetIdentity, batches: scene.batches.length, sources: scene.sources.length,
      firstEntityId, entities: last - offset, sourceInstances: scene.sourceInstances, pagedTriangles: scene.pagedTriangles, omittedTriangles: scene.omittedTriangles,
      mask: used, queryProxies: scene.queryProxies });
    result.sourceInstances += scene.sourceInstances;
    result.pagedTriangles += scene.pagedTriangles;
    result.omittedTriangles += scene.omittedTriangles;
    progress('component-ready', { component: component.name, data: { batches: scene.batches.length, textures: scene.sources.length } });
  }
  return result;
}
