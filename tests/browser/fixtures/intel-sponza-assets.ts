import { AnimationRig } from '../../../src/animation';
import { parseGltfNodeLayout } from '../../../src/gltf-node-layout';
import { createGltfMaterialLibrary } from '../../../src/gltf-material';
import { loadTextureStreamSource } from '../../../src/texture-stream-manifest';
import { validateGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { decodeGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import type { TextureStreamSource } from '../../../src/texture-stream-types';
import { normalMatrix, transformPoint } from '../../../src/math';
import type { StreamedSceneBatch } from '../../../src/contracts';

const MiB = 1024 ** 2;
async function checked(url: string, bytes: number, sha?: string, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok || !response.body) throw Error('SPONZA_HTTP: ' + url);
  const reader = response.body.getReader(), data = new Uint8Array(bytes); let offset = 0;
  try {
    while (true) {
      const next = await reader.read(); if (next.done) break;
      if (next.value.length > bytes - offset) throw Error('SPONZA_BYTE_BUDGET: ' + url);
      data.set(next.value, offset); offset += next.value.length;
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; }
  finally { reader.releaseLock(); }
  if (sha) {
    if (offset !== bytes) throw Error('SPONZA_BYTE_LENGTH: ' + url);
    const digest = await crypto.subtle.digest('SHA-256', data);
    if (Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('') !== sha) throw Error('SPONZA_HASH: ' + url);
  }
  return data.subarray(0, offset);
}
const json = async (url: string, limit: number) => JSON.parse(new TextDecoder().decode(await checked(url, limit)));

export interface IntelSponzaAssetOptions {
  geometryRoot?: string;
  queryRoot?: string;
  textureGpuRoot?: string;
  textureReferenceRoot?: string;
  uvAuditUrl?: string;
  sourceRaster?: boolean;
  anisotropy?: number;
}

/** Shared local asset loading for diagnostic fixtures. Hashes, source identity,
 * transforms, material conversion and explicit BLEND exclusions stay identical. */
export async function loadIntelSponzaAssets(settings: IntelSponzaAssetOptions = {},
  progress: (phase: string, data?: unknown) => void = () => {}) {
  const root = settings.geometryRoot ?? '/artifacts/sponza-group16-pages-20261022/';
  const textureGpuRoot = settings.textureGpuRoot ?? '/artifacts/sponza-textures-gpu-20261020/';
  const textureReferenceRoot = settings.textureReferenceRoot ?? '/artifacts/sponza-textures-reference-20261020/';
  const uvAuditUrl = settings.uvAuditUrl ?? '/artifacts/sponza-uv-demand-20261020.json';
  if (settings.sourceRaster && !settings.queryRoot) throw Error('SPONZA_SOURCE_RASTER_REQUIRES_FINE_QUERY');
  const manifest = await json(root + 'manifest.json', 32 * MiB);
  const queryManifest = settings.queryRoot ? await json(settings.queryRoot + 'manifest.json', 32 * MiB) : null;
  if (queryManifest && queryManifest.sourcePackage.assetIdentity !== manifest.assetIdentity) throw Error('SPONZA_FINE_PROXY_SOURCE');
  const queryEntries = new Map<string, any>((queryManifest?.primitives ?? []).map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p]));
  const doc = JSON.parse(new TextDecoder().decode(await checked(root + manifest.source.file, manifest.source.bytes, manifest.source.sha256)));
  if (doc.animations?.length || doc.skins?.length || doc.meshes.some((m: any) => m.primitives.some((p: any) => p.targets?.length))) throw Error('SPONZA_PREVIEW_EXPECTED_STATIC_SOURCE');
  const conversion = { path: 'materials[12].emissiveFactor[2]', before: doc.materials[12].emissiveFactor[2], after: 1 };
  if (Math.abs(conversion.before - 1) > 2 ** -22) throw Error('SPONZA_FACTOR_CONVERSION_BOUND');
  doc.materials[12].emissiveFactor[2] = 1;
  const layout = parseGltfNodeLayout(doc, doc.meshes.map(() => []));
  const pose = new AnimationRig(layout.rigNodes, []).sample(0);
  const roots = new Set(layout.scenes[layout.scene].roots);
  const nodesByMesh = new Map<number, number[]>();
  layout.nodes.forEach((node, i) => {
    let top = i; while (layout.nodes[top].parent !== null) top = layout.nodes[top].parent!;
    if (!roots.has(top) || node.mesh === undefined) return;
    const list = nodesByMesh.get(node.mesh) ?? []; list.push(i); nodesByMesh.set(node.mesh, list);
  });
  const textureManifest = await json(textureGpuRoot + 'manifest.json', MiB);
  const images = new Map(), sources: TextureStreamSource[] = [];
  for (const entry of textureManifest.images) {
    const reference = entry.roles.includes('normal:linear');
    const source = await loadTextureStreamSource((reference ? textureReferenceRoot : textureGpuRoot) + `image-${entry.image}/manifest.json`);
    images.set(entry.image, source.fallback); sources.push(source);
  }
  progress('textures-ready', { count: sources.length, referenceNormals: 23 });
  const materials = createGltfMaterialLibrary(doc, { images });
  const uvAudit = await json(uvAuditUrl, MiB);
  if (uvAudit.sourceSha256 !== manifest.source.sha256) throw Error('SPONZA_UV_SOURCE_MISMATCH');
  const uv = new Map(uvAudit.primitives.map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p.textureUvSpan]));
  const batches: StreamedSceneBatch[] = [], omitted: unknown[] = [], accounting = [];
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  let entityId = 1, sourceTriangles = 0, sourceInstances = 0;
  for (const entry of manifest.primitives) {
    const nodes = nodesByMesh.get(entry.meshIndex) ?? [];
    sourceInstances += nodes.length;
    if (entry.route !== 'paged') {
      omitted.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, material: entry.materialIndex,
        route: entry.route, coverage: entry.coverage, nodes, sourceTriangles: entry.sourceTriangleCount,
        reason: 'BLEND remains unsupported; explicit partial integration preview, not a completed scene' });
      continue;
    }
    if (!nodes.length) continue;
    const directory = validateGeometryStreamDirectory(JSON.parse(new TextDecoder().decode(await checked(root + entry.streamDirectoryFile,
      entry.streamDirectoryBytes, entry.streamDirectorySha256))));
    const proxyEntry = queryEntries.get(`${entry.meshIndex}:${entry.primitiveIndex}`)?.queryProxy ?? entry.queryProxy;
    if (settings.queryRoot && proxyEntry === entry.queryProxy) throw Error('SPONZA_FINE_PROXY_MISSING');
    const queryProxy = await decodeGeometryQueryProxy(directory, await checked((settings.queryRoot ?? root) + proxyEntry.file,
      proxyEntry.bytes, proxyEntry.sha256), proxyEntry.sha256,
      { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB });
    if (settings.sourceRaster && queryProxy.requestedError !== 0) throw Error('SPONZA_SOURCE_RASTER_REQUIRES_LOD0');
    if (settings.sourceRaster && queryProxy.mesh.indices.length / 3 !== entry.triangleCount) throw Error('SPONZA_SOURCE_RASTER_TRIANGLES');
    const verticesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].vertexCount, 0)));
    const indicesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].triangleCount * 3, 0)));
    const slotBytes = verticesPerSlot * 144 + indicesPerSlot * 4;
    const slots = Math.max(directory.rootPages.length, Math.ceil(directory.pages.length / 4));
    const gpuBytes = slots * slotBytes + directory.clusters.length * 32 + directory.pages.length * 4;
    const rootDecoded = directory.rootPages.reduce((n, id) => n + directory.pages[id].decodedBytes, 0);
    const largestPage = Math.max(...directory.pages.map(p => p.decodedBytes));
    const largestWork = Math.max(...directory.pages.map(p => 2 * p.encodedBytes + 2 * p.decodedBytes));
    const span = uv.get(`${entry.meshIndex}:${entry.primitiveIndex}`);
    const instances = nodes.map(node => {
      const model = pose.world[node];
      const determinant = model[0] * (model[5] * model[10] - model[9] * model[6]) - model[4] * (model[1] * model[10] - model[9] * model[2]) + model[8] * (model[1] * model[6] - model[5] * model[2]);
      if (determinant <= 0) throw Error('SPONZA_REFLECTED_INSTANCE: ' + node);
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
    const pageFiles = new Map(entry.pages.map((p: any) => [p.id, p]));
    const material = materials.get(entry.materialIndex);
    const previewMaterial = settings.anisotropy ? { ...material, samplers: material.textures?.map((_, slot) => {
      const sampler = material.samplers?.[slot];
      const allLinear = [sampler?.magFilter ?? 'linear', sampler?.minFilter ?? 'linear', sampler?.mipmapFilter ?? 'linear'].every(filter => filter === 'linear');
      return allLinear ? { ...sampler, maxAnisotropy: settings.anisotropy } : sampler;
    }) } : material;
    batches.push({ name: `sponza:${entry.meshIndex}:${entry.primitiveIndex}`, capacity: instances.length, instances,
      inputMode: 'committed', deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
      material: previewMaterial, ...(typeof span === 'number' ? { textureUvSpan: span } : {}),
      queryIdentity: { assetId: manifest.assetIdentity, primitiveId: batches.length, topologyVersion: 0 },
      geometryStream: { directory, queryProxy,
        ...(settings.queryRoot ? { queryProxyLimits: { maxVertices: 1048576, maxTriangles: 262144, maxBytes: 128 * MiB } } : {}),
        provider: async (page, signal) => { const p: any = pageFiles.get(page.id); return checked(root + p.file, page.encodedBytes, page.sha256, signal); },
        cpuCache: { maxResidentBytes: rootDecoded + 2 * largestPage, maxInFlightBytes: 2 * largestWork, maxConcurrentRequests: 2 },
        gpuPool: { maxResidentBytes: gpuBytes, maxStagingBytes: 2 * slotBytes + 65536,
          maxUploadBytesPerFrame: 256 * 1024, maxUploadPagesPerFrame: 1, maxQueuedPages: 8 }, maxRequestsPerFrame: 2 } });
    sourceTriangles += entry.triangleCount * instances.length;
    accounting.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, instances: instances.length, slots, pages: directory.pages.length, gpuBytes });
    if (batches.length % 50 === 0) progress('geometry-metadata', { batches: batches.length });
  }
  if (batches.length !== 401 || omitted.length !== 4) throw Error('SPONZA_CONTENT_COVERAGE');
  return { root, manifest, batches, accounting, bounds, sources, sourceInstances, sourceTriangles, conversion, omitted };
}
