/**
 * Loads an offline-built asset set (geometry pages, texture stream packages,
 * UV audit, optional exact query proxies; see scripts/write-asset-set.mjs)
 * into engine-owned scene batches. Every fetched geometry file is bounded and
 * SHA-256 checked against its manifest. Opaque paged primitives become
 * streamed batches. MASK primitives, which the package keeps source-routed,
 * are extracted from the hash-checked source buffer into full-detail
 * (non-paged) batches within an explicit budget, so the main view, shadows and
 * world queries all draw them. BLEND primitives and mirrored instances are
 * returned in `omitted` with their triangle counts, never silently dropped or
 * relabelled.
 */
import { AnimationRig } from './animation';
import type { InstanceState, SceneBatch, StreamedSceneBatch } from './contracts';
import { extractGltfStaticPrimitive } from './gltf-geometry-build';
import { decodeGeometryQueryProxy } from './geometry-query-proxy';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import { createGltfMaterialLibrary } from './gltf-material';
import { parseGltfNodeLayout } from './gltf-node-layout';
import { normalMatrix, transformPoint } from './math';
import { loadTextureStreamSource } from './texture-stream-manifest';
import type { TextureStreamSource } from './texture-stream-types';

const MiB = 1024 ** 2;
const PROXY_LIMITS = { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB };
/** Page requests, decodes and uploads each streamed batch may start per frame. */
const STREAM_PAGES_PER_FRAME = 4;
/** float32 rounding headroom: factors at most this far above 1 are clamped (and recorded). */
const FACTOR_EPSILON = 2 ** -22;

export interface GltfPackageRoots {
  geometryRoot: string;
  textureGpuRoot: string;
  textureReferenceRoot: string;
  uvAuditUrl: string;
  cameraPathUrl?: string;
  /** Optional package of exact (or finer) query proxies built from the same geometry package. */
  fineQueryRoot?: string;
  /** Package manifest hashes recorded by the asset set; each fetched manifest must match. */
  manifestSha256?: { geometry?: string; textures?: string; fineQuery?: string };
}

/** Source-geometry budget of the full-detail MASK batches (per source primitive, not per instance). */
export interface GltfPackageMaskBudget {
  /** Default 1024. */
  maxPrimitives?: number;
  /** Default 4,194,304. */
  maxVertices?: number;
  /** Default 2,097,152. */
  maxTriangles?: number;
}

export interface GltfPackageSceneOptions {
  /** Batch name prefix, e.g. 'bistro'. */
  name?: string;
  anisotropy?: number;
  maxBatches?: number;
  signal?: AbortSignal;
  progress?: (phase: string, data?: unknown) => void;
  /** MASK primitives are drawn full-detail within this budget (default); false omits and reports them. */
  mask?: GltfPackageMaskBudget | false;
}

export interface OmittedPrimitive {
  mesh: number; primitive: number; material: number | null; route: string; coverage: string;
  nodes: number[]; sourceTriangles: number; reason: string;
}

export interface GltfPackageMaskSummary {
  primitives: number;
  sourceVertices: number;
  sourceTriangles: number;
  /** Instance-expanded triangles of the drawn MASK batches. */
  triangles: number;
  /** CPU mesh bytes (interleaved vertices, attributes, indices) handed to the renderer. */
  meshBytes: number;
  /** Hash-checked source buffer bytes fetched to extract them (0 when none). */
  sourceBufferBytes: number;
  budget: Required<GltfPackageMaskBudget> | null;
}

export interface GltfPackageScene {
  /** Streamed (paged, opaque) batches and full-detail MASK batches, in package order. */
  batches: (StreamedSceneBatch | SceneBatch)[];
  sources: TextureStreamSource[];
  bounds: { min: [number, number, number]; max: [number, number, number] };
  omitted: OmittedPrimitive[];
  conversions: { path: string; before: number; after: number }[];
  /** Material inputs present in the asset that the engine material model does not read. */
  unsupportedMaterialInputs: { path: string; reason: string }[];
  sourceInstances: number;
  /** Instance-expanded triangles of the streamed (paged) batches. */
  pagedTriangles: number;
  /** Instance-expanded triangles of omitted primitives. */
  omittedTriangles: number;
  mask: GltfPackageMaskSummary;
  /** Shadow/world-query representation of the streamed batches. */
  queryProxies: { root: string; fine: boolean; maxError: number | null; primitives: number; triangles: number };
  assetIdentity: string;
}

function localPath(value: unknown, label: string, directory: boolean): string {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || value.includes('..')
    || (directory && !value.endsWith('/'))) throw new Error('GLTF_PACKAGE_ROOT: ' + label);
  return value;
}

async function checked(url: string, bytes: number, sha?: string, signal?: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  if (!Number.isSafeInteger(bytes) || bytes < 1 || bytes > 1024 * MiB) throw new Error('GLTF_PACKAGE_BYTE_BUDGET: ' + url);
  const response = await fetch(url, { signal });
  if (!response.ok || !response.body) throw new Error('GLTF_PACKAGE_HTTP: ' + url);
  const reader = response.body.getReader(), data = new Uint8Array(bytes);
  let offset = 0;
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      if (next.value.length > bytes - offset) throw new Error('GLTF_PACKAGE_BYTE_BUDGET: ' + url);
      data.set(next.value, offset); offset += next.value.length;
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; } finally { reader.releaseLock(); }
  if (sha) {
    if (offset !== bytes) throw new Error('GLTF_PACKAGE_BYTE_LENGTH: ' + url);
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data));
    if (Array.from(digest, n => n.toString(16).padStart(2, '0')).join('') !== sha) throw new Error('GLTF_PACKAGE_HASH: ' + url);
  }
  return data.subarray(0, offset);
}

function budget(value: number | undefined, fallback: number, maximum: number, name: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 0 || result > maximum) throw new Error('GLTF_PACKAGE_MASK_OPTION: ' + name);
  return result;
}

/** Bounded JSON fetch; with `sha256`, the bytes read must hash to it (manifests whose length is not recorded). */
async function json(url: string, limit: number, signal?: AbortSignal, sha256?: string) {
  const bytes = await checked(url, limit, undefined, signal);
  if (sha256 !== undefined) {
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), n => n.toString(16).padStart(2, '0')).join('');
    if (digest !== sha256) throw new Error('GLTF_PACKAGE_MANIFEST_HASH: ' + url);
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

const sha256Of = (value: unknown, label: string): string | undefined => {
  if (value === undefined) return undefined;
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) throw new Error('GLTF_PACKAGE_ASSET_SET_HASH: ' + label);
  return value;
};

/** Reads asset-set.json (format real-engine-asset-set) and returns its local server roots and recorded manifest hashes. */
export async function loadAssetSetRoots(url: string, signal?: AbortSignal): Promise<GltfPackageRoots> {
  const set = await json(localPath(url, 'assetSet', false), MiB, signal);
  if (set?.format !== 'real-engine-asset-set' || set.version !== 1 || !set.roots) throw new Error('GLTF_PACKAGE_ASSET_SET_FORMAT');
  const roots: GltfPackageRoots = {
    geometryRoot: localPath(set.roots.geometryRoot, 'geometryRoot', true),
    textureGpuRoot: localPath(set.roots.textureGpuRoot, 'textureGpuRoot', true),
    textureReferenceRoot: localPath(set.roots.textureReferenceRoot, 'textureReferenceRoot', true),
    uvAuditUrl: localPath(set.roots.uvAuditUrl, 'uvAuditUrl', false),
  };
  if (set.roots.cameraPathUrl !== undefined) roots.cameraPathUrl = localPath(set.roots.cameraPathUrl, 'cameraPathUrl', false);
  if (set.roots.fineQueryRoot !== undefined) roots.fineQueryRoot = localPath(set.roots.fineQueryRoot, 'fineQueryRoot', true);
  const hashes = { geometry: sha256Of(set.packages?.geometry?.manifestSha256, 'geometry'),
    textures: sha256Of(set.packages?.textures?.manifestSha256, 'textures'), fineQuery: sha256Of(set.packages?.fineQuery?.manifestSha256, 'fineQuery') };
  if (hashes.geometry || hashes.textures || hashes.fineQuery) {
    roots.manifestSha256 = Object.fromEntries(Object.entries(hashes).filter(([, value]) => value !== undefined));
  }
  return roots;
}

/** Clamps material factors that exceed 1 only by float32 rounding; returns the edits. */
export function clampMaterialFactors(doc: { materials?: Record<string, any>[] }): { path: string; before: number; after: number }[] {
  const conversions: { path: string; before: number; after: number }[] = [];
  const visit = (values: unknown, path: string, set: (value: number) => void) => {
    if (typeof values !== 'number' || values <= 1) return;
    if (values > 1 + FACTOR_EPSILON) throw new Error('GLTF_PACKAGE_FACTOR_RANGE: ' + path);
    conversions.push({ path, before: values, after: 1 }); set(1);
  };
  (doc.materials ?? []).forEach((material, m) => {
    const pbr = material.pbrMetallicRoughness ?? {};
    (pbr.baseColorFactor ?? []).forEach((v: number, i: number) => visit(v, `materials[${m}].pbrMetallicRoughness.baseColorFactor[${i}]`, x => { pbr.baseColorFactor[i] = x; }));
    for (const key of ['metallicFactor', 'roughnessFactor']) visit(pbr[key], `materials[${m}].pbrMetallicRoughness.${key}`, x => { pbr[key] = x; });
    (material.emissiveFactor ?? []).forEach((v: number, i: number) => visit(v, `materials[${m}].emissiveFactor[${i}]`, x => { material.emissiveFactor[i] = x; }));
  });
  return conversions;
}

export async function loadGltfPackageScene(roots: GltfPackageRoots, options: GltfPackageSceneOptions = {}): Promise<GltfPackageScene> {
  const { signal, progress = () => {} } = options, prefix = options.name ?? 'scene';
  const root = localPath(roots.geometryRoot, 'geometryRoot', true);
  const manifest = await json(root + 'manifest.json', 64 * MiB, signal, roots.manifestSha256?.geometry);
  const doc = JSON.parse(new TextDecoder().decode(await checked(root + manifest.source.file, manifest.source.bytes, manifest.source.sha256, signal)));
  if (doc.animations?.length || doc.skins?.length || doc.meshes.some((m: any) => m.primitives.some((p: any) => p.targets?.length)))
    throw new Error('GLTF_PACKAGE_EXPECTED_STATIC_SOURCE');
  const conversions = clampMaterialFactors(doc);
  // The engine material model has no ambient-occlusion input; the texture stays
  // in the package (and in shared packed images) but is reported, not applied.
  const unsupportedMaterialInputs: { path: string; reason: string }[] = [];
  (doc.materials ?? []).forEach((material: Record<string, unknown>, m: number) => {
    if (material.occlusionTexture === undefined) return;
    delete material.occlusionTexture;
    unsupportedMaterialInputs.push({ path: `materials[${m}].occlusionTexture`, reason: 'occlusion textures are not read by the engine material model' });
  });
  const layout = parseGltfNodeLayout(doc, doc.meshes.map(() => []));
  const pose = new AnimationRig(layout.rigNodes, []).sample(0);
  const sceneRoots = new Set(layout.scenes[layout.scene].roots);
  const nodesByMesh = new Map<number, number[]>();
  layout.nodes.forEach((node, i) => {
    let top = i;
    while (layout.nodes[top].parent !== null) top = layout.nodes[top].parent!;
    if (!sceneRoots.has(top) || node.mesh === undefined) return;
    const list = nodesByMesh.get(node.mesh) ?? [];
    list.push(i); nodesByMesh.set(node.mesh, list);
  });
  const textureManifest = await json(roots.textureGpuRoot + 'manifest.json', 4 * MiB, signal, roots.manifestSha256?.textures);
  if (textureManifest.format !== 'real-engine-gltf-texture-set' || textureManifest.source?.sha256 !== manifest.source.sha256)
    throw new Error('GLTF_PACKAGE_TEXTURE_SOURCE');
  const images = new Map(), sources: TextureStreamSource[] = [];
  for (const entry of textureManifest.images) {
    const reference = entry.path === 'reference';
    const source = await loadTextureStreamSource((reference ? roots.textureReferenceRoot : roots.textureGpuRoot) + `image-${entry.image}/manifest.json`, { signal });
    images.set(entry.image, source.fallback); sources.push(source);
  }
  progress('textures-ready', { count: sources.length });
  const materials = createGltfMaterialLibrary(doc, { images });
  const uvAudit = await json(roots.uvAuditUrl, 8 * MiB, signal);
  if (uvAudit.sourceSha256 !== manifest.source.sha256) throw new Error('GLTF_PACKAGE_UV_SOURCE');
  const uv = new Map(uvAudit.primitives.map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p.textureUvSpan]));
  const fine = roots.fineQueryRoot ? await json(localPath(roots.fineQueryRoot, 'fineQueryRoot', true) + 'manifest.json', 64 * MiB, signal, roots.manifestSha256?.fineQuery) : null;
  if (fine && (fine.format !== 'real-engine-gltf-query-proxies' || fine.sourcePackage?.assetIdentity !== manifest.assetIdentity
    || fine.sourcePackage?.sourceSha256 !== manifest.source.sha256)) throw new Error('GLTF_PACKAGE_QUERY_SOURCE');
  const fineEntries = new Map<string, any>((fine?.primitives ?? []).map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p]));
  const maskBudget = options.mask === false ? null : {
    maxPrimitives: budget(options.mask?.maxPrimitives, 1024, 4096, 'maxPrimitives'),
    maxVertices: budget(options.mask?.maxVertices, 4_194_304, 16_777_216, 'maxVertices'),
    maxTriangles: budget(options.mask?.maxTriangles, 2_097_152, 16_777_216, 'maxTriangles') };
  const maskEntries = manifest.primitives.filter((entry: any) => entry.route !== 'paged' && entry.coverage === 'mask'
    && (nodesByMesh.get(entry.meshIndex) ?? []).length);
  const mask: GltfPackageMaskSummary = { primitives: 0, sourceVertices: 0, sourceTriangles: 0, triangles: 0, meshBytes: 0, sourceBufferBytes: 0, budget: maskBudget };
  if (maskBudget) {
    const vertices = maskEntries.reduce((n: number, e: any) => n + e.sourceVertexCount, 0);
    const triangles = maskEntries.reduce((n: number, e: any) => n + e.sourceTriangleCount, 0);
    if (maskEntries.length > maskBudget.maxPrimitives || vertices > maskBudget.maxVertices || triangles > maskBudget.maxTriangles) {
      throw new Error(`GLTF_PACKAGE_MASK_BUDGET: ${maskEntries.length} primitives, ${vertices} vertices, ${triangles} triangles exceed ${JSON.stringify(maskBudget)}`);
    }
  }
  // MASK geometry is read once from the package's hash-checked copy of the glTF buffers, before any batch exists.
  let gltfBuffers: ArrayBuffer[] | null = null;
  if (maskBudget && maskEntries.length) {
    const declared = manifest.source.buffers;
    if (!Array.isArray(declared) || declared.length !== (doc.buffers ?? []).length) throw new Error('GLTF_PACKAGE_SOURCE_BUFFERS');
    gltfBuffers = [];
    for (const [index, buffer] of declared.entries()) {
      if (buffer.index !== index || buffer.embedded !== false || typeof buffer.file !== 'string' || buffer.file.includes('..')) throw new Error('GLTF_PACKAGE_SOURCE_BUFFERS');
      gltfBuffers.push((await checked(root + buffer.file, buffer.bytes, buffer.sha256, signal)).buffer);
      mask.sourceBufferBytes += buffer.bytes;
    }
    progress('mask-buffers-ready', { bytes: mask.sourceBufferBytes });
  }
  const batches: (StreamedSceneBatch | SceneBatch)[] = [], omitted: OmittedPrimitive[] = [];
  const bounds = { min: [Infinity, Infinity, Infinity] as [number, number, number], max: [-Infinity, -Infinity, -Infinity] as [number, number, number] };
  let entityId = 1, sourceInstances = 0, pagedTriangles = 0, omittedTriangles = 0, proxyTriangles = 0, proxies = 0;
  const maxBatches = options.maxBatches ?? 4096;
  const instancesOf = (drawn: number[], local: { min: readonly number[]; max: readonly number[] }[]): InstanceState[] => drawn.map(node => {
    const model = pose.world[node];
    for (const b of local) for (let corner = 0; corner < 8; corner++) {
      const p = transformPoint(model, [b[(corner & 1) ? 'max' : 'min'][0], b[(corner & 2) ? 'max' : 'min'][1], b[(corner & 4) ? 'max' : 'min'][2]]);
      for (let axis = 0; axis < 3; axis++) { bounds.min[axis] = Math.min(bounds.min[axis], p[axis]); bounds.max[axis] = Math.max(bounds.max[axis], p[axis]); }
    }
    return { entityId: entityId++, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
      color: [1, 1, 1, 1] as [number, number, number, number], metalness: 1, roughness: 1, historyVersion: 0 };
  });
  const materialOf = (index: number | null) => {
    const material = materials.get(index);
    return options.anisotropy && material ? { ...material, samplers: material.textures?.map((_, slot) => {
      const sampler = material.samplers?.[slot];
      const allLinear = [sampler?.magFilter ?? 'linear', sampler?.minFilter ?? 'linear', sampler?.mipmapFilter ?? 'linear'].every(f => f === 'linear');
      return allLinear ? { ...sampler, maxAnisotropy: options.anisotropy } : sampler;
    }) } : material;
  };
  for (const entry of manifest.primitives) {
    const nodes = nodesByMesh.get(entry.meshIndex) ?? [];
    sourceInstances += nodes.length;
    const omit = (reason: string, list: number[]) => {
      omitted.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, material: entry.materialIndex, route: entry.route,
        coverage: entry.coverage, nodes: list, sourceTriangles: entry.sourceTriangleCount, reason });
      omittedTriangles += entry.sourceTriangleCount * list.length;
    };
    const fullDetailMask = entry.route !== 'paged' && entry.coverage === 'mask' && !!maskBudget;
    if (entry.route !== 'paged' && !fullDetailMask) {
      omit(entry.coverage === 'blend' ? 'BLEND coverage is not supported by the renderer; omitted, never drawn as MASK or opaque'
        : `${entry.coverage} coverage is source-routed and the full-detail MASK path is disabled`, nodes);
      continue;
    }
    // Mirrored transforms flip winding and the renderer does not swap the front face for them: reported, never drawn inside out.
    const mirrored = nodes.filter(node => {
      const m = pose.world[node];
      return m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]) <= 0;
    });
    if (mirrored.length) omit('mirrored instance transform (non-positive determinant) is not supported', mirrored);
    const drawn = nodes.filter(node => !mirrored.includes(node));
    if (!drawn.length) continue;
    if (batches.length >= maxBatches) throw new Error('GLTF_PACKAGE_BATCH_BUDGET');
    const key = `${entry.meshIndex}:${entry.primitiveIndex}`, span = uv.get(key);
    const common = { name: `${prefix}:${key}`, inputMode: 'committed' as const, deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
      material: materialOf(entry.materialIndex), ...(typeof span === 'number' ? { textureUvSpan: span } : {}),
      queryIdentity: { assetId: manifest.assetIdentity, primitiveId: batches.length, topologyVersion: 0 } };
    if (fullDetailMask) {
      const extracted = extractGltfStaticPrimitive(doc, gltfBuffers!, entry.meshIndex, entry.primitiveIndex), mesh = extracted.mesh;
      if (extracted.coverage !== 'mask' || mesh.vertexCount !== entry.sourceVertexCount || mesh.indices.length !== entry.sourceTriangleCount * 3) {
        throw new Error('GLTF_PACKAGE_MASK_SOURCE_MISMATCH: ' + key);
      }
      const instances = instancesOf(drawn, [mesh.bounds]);
      batches.push({ ...common, capacity: instances.length, instances, mesh });
      mask.primitives++; mask.sourceVertices += mesh.vertexCount; mask.sourceTriangles += entry.sourceTriangleCount;
      mask.triangles += entry.sourceTriangleCount * instances.length;
      mask.meshBytes += mesh.vertices.byteLength + mesh.indices.byteLength + (mesh.uvs?.byteLength ?? 0) + (mesh.tangents?.byteLength ?? 0) + (mesh.colors?.byteLength ?? 0);
      continue;
    }
    const directory = validateGeometryStreamDirectory(JSON.parse(new TextDecoder().decode(await checked(root + entry.streamDirectoryFile,
      entry.streamDirectoryBytes, entry.streamDirectorySha256, signal))));
    const fineEntry = fineEntries.get(key);
    if (fine && (!fineEntry || fineEntry.streamDirectorySha256 !== entry.streamDirectorySha256)) throw new Error('GLTF_PACKAGE_QUERY_MISSING: ' + key);
    const proxyEntry = fineEntry?.queryProxy ?? entry.queryProxy, proxyRoot = fineEntry ? roots.fineQueryRoot! : root;
    const queryProxy = await decodeGeometryQueryProxy(directory, await checked(proxyRoot + proxyEntry.file, proxyEntry.bytes,
      proxyEntry.sha256, signal), proxyEntry.sha256, PROXY_LIMITS);
    proxies++; proxyTriangles += queryProxy.mesh.indices.length / 3;
    const verticesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].vertexCount, 0)));
    const indicesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].triangleCount * 3, 0)));
    const slotBytes = verticesPerSlot * 144 + indicesPerSlot * 4;
    // Default initial pool: a quarter of the pages. Callers with a scene-wide budget replace it from their plan.
    const slots = Math.max(directory.rootPages.length, Math.ceil(directory.pages.length / 4));
    const gpuBytes = slots * slotBytes + directory.clusters.length * 32 + directory.pages.length * 4;
    const rootDecoded = directory.rootPages.reduce((n, id) => n + directory.pages[id].decodedBytes, 0);
    const largestPage = Math.max(...directory.pages.map(p => p.decodedBytes));
    const largestWork = Math.max(...directory.pages.map(p => 2 * p.encodedBytes + 2 * p.decodedBytes));
    const instances = instancesOf(drawn, directory.roots.map(id => directory.clusters[id].lodBounds));
    const pageFiles = new Map<number, { file: string }>(entry.pages.map((p: { id: number; file: string }) => [p.id, p]));
    batches.push({ ...common, capacity: instances.length, instances,
      geometryStream: { directory, queryProxy, queryProxyLimits: PROXY_LIMITS,
        provider: async (page, pageSignal) => checked(root + pageFiles.get(page.id)!.file, page.encodedBytes, page.sha256, pageSignal),
        // Per stream: up to STREAM_PAGES_PER_FRAME pages decoded, staged and uploaded per frame, each payload bounded.
        cpuCache: { maxResidentBytes: rootDecoded + STREAM_PAGES_PER_FRAME * largestPage, maxInFlightBytes: STREAM_PAGES_PER_FRAME * largestWork,
          maxConcurrentRequests: STREAM_PAGES_PER_FRAME },
        gpuPool: { maxResidentBytes: gpuBytes, maxStagingBytes: STREAM_PAGES_PER_FRAME * slotBytes + 65536,
          maxUploadBytesPerFrame: 1024 * 1024, maxUploadPagesPerFrame: STREAM_PAGES_PER_FRAME, maxQueuedPages: 4 * STREAM_PAGES_PER_FRAME },
        maxRequestsPerFrame: STREAM_PAGES_PER_FRAME } });
    pagedTriangles += entry.triangleCount * instances.length;
    if (batches.length % 100 === 0) progress('geometry-metadata', { batches: batches.length });
  }
  return { batches, sources, bounds, omitted, conversions, unsupportedMaterialInputs, sourceInstances, pagedTriangles, omittedTriangles, mask,
    queryProxies: { root: fine ? roots.fineQueryRoot! : root, fine: !!fine, maxError: fine ? Number(fine.config?.maxError) : null,
      primitives: proxies, triangles: proxyTriangles },
    assetIdentity: manifest.assetIdentity };
}
