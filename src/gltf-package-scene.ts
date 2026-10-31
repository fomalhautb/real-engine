/**
 * Loads an offline-built asset set (geometry pages, texture stream packages,
 * UV audit; see scripts/write-asset-set.mjs) into engine-owned scene batches.
 * Every fetched geometry file is bounded and SHA-256 checked against its
 * manifest. Only opaque paged primitives become streamed batches: MASK/BLEND
 * primitives stay source-routed in the package and are returned in `omitted`
 * with their triangle counts, never silently dropped or relabelled opaque.
 */
import { AnimationRig } from './animation';
import type { StreamedSceneBatch } from './contracts';
import { decodeGeometryQueryProxy } from './geometry-query-proxy';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import { createGltfMaterialLibrary } from './gltf-material';
import { parseGltfNodeLayout } from './gltf-node-layout';
import { normalMatrix, transformPoint } from './math';
import { loadTextureStreamSource } from './texture-stream-manifest';
import type { TextureStreamSource } from './texture-stream-types';

const MiB = 1024 ** 2;
const PROXY_LIMITS = { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB };
/** float32 rounding headroom: factors at most this far above 1 are clamped (and recorded). */
const FACTOR_EPSILON = 2 ** -22;

export interface GltfPackageRoots {
  geometryRoot: string;
  textureGpuRoot: string;
  textureReferenceRoot: string;
  uvAuditUrl: string;
  cameraPathUrl?: string;
}

export interface GltfPackageSceneOptions {
  /** Batch name prefix, e.g. 'bistro'. */
  name?: string;
  anisotropy?: number;
  maxBatches?: number;
  signal?: AbortSignal;
  progress?: (phase: string, data?: unknown) => void;
}

export interface OmittedPrimitive {
  mesh: number; primitive: number; material: number | null; route: string; coverage: string;
  nodes: number[]; sourceTriangles: number; reason: string;
}

export interface GltfPackageScene {
  batches: StreamedSceneBatch[];
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

const json = async (url: string, limit: number, signal?: AbortSignal) => JSON.parse(new TextDecoder().decode(await checked(url, limit, undefined, signal)));

/** Reads asset-set.json (format real-engine-asset-set) and returns its local server roots. */
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
  const manifest = await json(root + 'manifest.json', 64 * MiB, signal);
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
  const textureManifest = await json(roots.textureGpuRoot + 'manifest.json', 4 * MiB, signal);
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
  const batches: StreamedSceneBatch[] = [], omitted: OmittedPrimitive[] = [];
  const bounds = { min: [Infinity, Infinity, Infinity] as [number, number, number], max: [-Infinity, -Infinity, -Infinity] as [number, number, number] };
  let entityId = 1, sourceInstances = 0, pagedTriangles = 0, omittedTriangles = 0;
  const maxBatches = options.maxBatches ?? 4096;
  for (const entry of manifest.primitives) {
    const nodes = nodesByMesh.get(entry.meshIndex) ?? [];
    sourceInstances += nodes.length;
    const omit = (reason: string, list: number[]) => {
      omitted.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, material: entry.materialIndex, route: entry.route,
        coverage: entry.coverage, nodes: list, sourceTriangles: entry.sourceTriangleCount, reason });
      omittedTriangles += entry.sourceTriangleCount * list.length;
    };
    if (entry.route !== 'paged') { omit(`${entry.coverage} coverage is source-routed; virtual geometry requires opaque coverage`, nodes); continue; }
    // Mirrored transforms would flip winding; they are reported, not drawn inside out.
    const mirrored = nodes.filter(node => {
      const m = pose.world[node];
      return m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]) <= 0;
    });
    if (mirrored.length) omit('mirrored instance transform (non-positive determinant) is not supported', mirrored);
    const drawn = nodes.filter(node => !mirrored.includes(node));
    if (!drawn.length) continue;
    if (batches.length >= maxBatches) throw new Error('GLTF_PACKAGE_BATCH_BUDGET');
    const directory = validateGeometryStreamDirectory(JSON.parse(new TextDecoder().decode(await checked(root + entry.streamDirectoryFile,
      entry.streamDirectoryBytes, entry.streamDirectorySha256, signal))));
    const queryProxy = await decodeGeometryQueryProxy(directory, await checked(root + entry.queryProxy.file, entry.queryProxy.bytes,
      entry.queryProxy.sha256, signal), entry.queryProxy.sha256, PROXY_LIMITS);
    const verticesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].vertexCount, 0)));
    const indicesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].triangleCount * 3, 0)));
    const slotBytes = verticesPerSlot * 144 + indicesPerSlot * 4;
    const slots = Math.max(directory.rootPages.length, Math.ceil(directory.pages.length / 4));
    const gpuBytes = slots * slotBytes + directory.clusters.length * 32 + directory.pages.length * 4;
    const rootDecoded = directory.rootPages.reduce((n, id) => n + directory.pages[id].decodedBytes, 0);
    const largestPage = Math.max(...directory.pages.map(p => p.decodedBytes));
    const largestWork = Math.max(...directory.pages.map(p => 2 * p.encodedBytes + 2 * p.decodedBytes));
    const instances = drawn.map(node => {
      const model = pose.world[node];
      for (const id of directory.roots) {
        const b = directory.clusters[id].lodBounds;
        for (let corner = 0; corner < 8; corner++) {
          const p = transformPoint(model, [b[(corner & 1) ? 'max' : 'min'][0], b[(corner & 2) ? 'max' : 'min'][1], b[(corner & 4) ? 'max' : 'min'][2]]);
          for (let axis = 0; axis < 3; axis++) { bounds.min[axis] = Math.min(bounds.min[axis], p[axis]); bounds.max[axis] = Math.max(bounds.max[axis], p[axis]); }
        }
      }
      return { entityId: entityId++, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
        color: [1, 1, 1, 1] as [number, number, number, number], metalness: 1, roughness: 1, historyVersion: 0 };
    });
    const pageFiles = new Map<number, { file: string }>(entry.pages.map((p: { id: number; file: string }) => [p.id, p]));
    const material = materials.get(entry.materialIndex);
    const anisotropic = options.anisotropy && material ? { ...material, samplers: material.textures?.map((_, slot) => {
      const sampler = material.samplers?.[slot];
      const allLinear = [sampler?.magFilter ?? 'linear', sampler?.minFilter ?? 'linear', sampler?.mipmapFilter ?? 'linear'].every(f => f === 'linear');
      return allLinear ? { ...sampler, maxAnisotropy: options.anisotropy } : sampler;
    }) } : material;
    const span = uv.get(`${entry.meshIndex}:${entry.primitiveIndex}`);
    batches.push({ name: `${prefix}:${entry.meshIndex}:${entry.primitiveIndex}`, capacity: instances.length, instances,
      inputMode: 'committed', deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: anisotropic,
      ...(typeof span === 'number' ? { textureUvSpan: span } : {}),
      queryIdentity: { assetId: manifest.assetIdentity, primitiveId: batches.length, topologyVersion: 0 },
      geometryStream: { directory, queryProxy,
        provider: async (page, pageSignal) => checked(root + pageFiles.get(page.id)!.file, page.encodedBytes, page.sha256, pageSignal),
        cpuCache: { maxResidentBytes: rootDecoded + 2 * largestPage, maxInFlightBytes: 2 * largestWork, maxConcurrentRequests: 2 },
        gpuPool: { maxResidentBytes: gpuBytes, maxStagingBytes: 2 * slotBytes + 65536,
          maxUploadBytesPerFrame: 256 * 1024, maxUploadPagesPerFrame: 1, maxQueuedPages: 8 }, maxRequestsPerFrame: 2 } });
    pagedTriangles += entry.triangleCount * instances.length;
    if (batches.length % 100 === 0) progress('geometry-metadata', { batches: batches.length });
  }
  return { batches, sources, bounds, omitted, conversions, unsupportedMaterialInputs, sourceInstances, pagedTriangles, omittedTriangles,
    assetIdentity: manifest.assetIdentity };
}
