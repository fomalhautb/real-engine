import { parseGlb } from './gltf-binary';
import { importGltf, prepareGltfTangents, type GltfAsset, type GltfImportOptions } from './gltf-import';
import { decodeTexture, type TextureAsset } from './texture-assets';

export interface GltfLoadOptions extends Omit<GltfImportOptions, 'images'> {
  signal?: AbortSignal;
  /** Cumulative compressed/downloaded bytes, including JSON, buffers and images. */
  maxBytes?: number;
  /** Cumulative decoded RGBA8 image bytes, separate from the download budget. */
  maxDecodedImageBytes?: number;
  fetch?: typeof fetch;
}

/** A bounded hint for async preparation; importGltf performs the authoritative validation. */
function mayNeedGeneratedTangents(json: Record<string, unknown>): boolean {
  const meshes = json.meshes, materials = json.materials;
  if (!Array.isArray(meshes) || meshes.length > 65536 || !Array.isArray(materials) || materials.length > 65536) return false;
  if (!materials.some(value => value && typeof value === 'object' && !Array.isArray(value)
    && (value as Record<string, unknown>).normalTexture !== undefined)) return false;
  let inspected = 0;
  for (const mesh of meshes) {
    if (!mesh || typeof mesh !== 'object' || Array.isArray(mesh)) continue;
    const primitives: unknown = (mesh as Record<string, unknown>).primitives;
    if (!Array.isArray(primitives) || primitives.length > 4096) continue;
    for (const value of primitives) {
      if (++inspected > 1_000_000) return true;
      if (!value || typeof value !== 'object' || Array.isArray(value)) continue;
      const primitive = value as Record<string, unknown>;
      const materialIndex = primitive.material;
      if (typeof materialIndex !== 'number' || !Number.isSafeInteger(materialIndex) || materialIndex < 0 || materialIndex >= materials.length) continue;
      const material = materials[materialIndex];
      if (!material || typeof material !== 'object' || Array.isArray(material) || (material as Record<string, unknown>).normalTexture === undefined) continue;
      const attrs = primitive.attributes;
      if (!attrs || typeof attrs !== 'object' || Array.isArray(attrs)) continue;
      const attributes = attrs as Record<string, unknown>;
      if (attributes.NORMAL === undefined || attributes.TANGENT === undefined) return true;
    }
  }
  return false;
}

/** Bounded network loading. Import and texture decoding remain explicit CPU work for now. */
export async function loadGltf(url: string | URL, options: GltfLoadOptions = {}): Promise<GltfAsset> {
  const base = new URL(url, typeof location === 'undefined' ? undefined : location.href);
  if (!['https:', 'http:'].includes(base.protocol)) throw new Error('GLTF_URL_PROTOCOL');
  const maxBytes = options.maxBytes ?? 128 * 1024 * 1024;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new Error('GLTF_LOAD_BUDGET');
  const maxDecodedImageBytes = options.maxDecodedImageBytes ?? 256 * 1024 * 1024;
  if (!Number.isSafeInteger(maxDecodedImageBytes) || maxDecodedImageBytes < 0) throw new Error('GLTF_DECODED_IMAGE_BUDGET');
  const fetcher = options.fetch ?? fetch;
  let downloaded = 0;
  const charge = (bytes: number) => { downloaded += bytes; if (downloaded > maxBytes) throw new Error('GLTF_DOWNLOAD_BUDGET'); };
  const read = async (target: URL): Promise<ArrayBuffer> => {
    options.signal?.throwIfAborted();
    if (!['https:', 'http:', 'data:'].includes(target.protocol)) throw new Error('GLTF_RESOURCE_URL_PROTOCOL');
    const response = await fetcher(target, { signal: options.signal, credentials: 'omit' });
    if (!response.ok) throw new Error('GLTF_FETCH: ' + response.status + ' ' + target.pathname);
    const length = response.headers.get('content-length');
    if (length !== null && Number(length) > maxBytes - downloaded) { await response.body?.cancel(); throw new Error('GLTF_DOWNLOAD_BUDGET'); }
    if (!response.body) throw new Error('GLTF_EMPTY_RESPONSE');
    const reader = response.body.getReader(), chunks: Uint8Array[] = [];
    let total = 0;
    try {
      for (;;) {
        options.signal?.throwIfAborted();
        const { done, value } = await reader.read();
        if (done) break;
        charge(value.byteLength); total += value.byteLength; chunks.push(value);
      }
    } catch (error) { await reader.cancel().catch(() => {}); throw error; }
    finally { reader.releaseLock(); }
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return bytes.buffer;
  };
  const data = await read(base);
  const container = data.byteLength >= 4 && new DataView(data).getUint32(0, true) === 0x46546c67
    ? parseGlb(data) : { json: JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(data)) as Record<string, unknown>, bin: undefined };
  const json = container.json;
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error('GLTF_ROOT_OBJECT');
  if (mayNeedGeneratedTangents(json)) {
    options.signal?.throwIfAborted();
    await prepareGltfTangents();
    options.signal?.throwIfAborted();
  }
  const list = (value: unknown, name: string): Record<string, unknown>[] => {
    if (value === undefined) return [];
    if (!Array.isArray(value) || value.length > 4096 || value.some(item => !item || typeof item !== 'object' || Array.isArray(item))) throw new Error('GLTF_RESOURCE_LIST: ' + name);
    return value as Record<string, unknown>[];
  };
  // Sequential downloads bound outstanding allocations and fail before requesting the next resource.
  const bufferDefinitions = list(json.buffers, 'buffers');
  const buffers: ArrayBuffer[] = [];
  for (const [i, buffer] of bufferDefinitions.entries()) {
    if (!Number.isSafeInteger(buffer.byteLength) || (buffer.byteLength as number) < 0) throw new Error('GLTF_BUFFER_LENGTH');
    const bytes = typeof buffer.uri === 'string' ? await read(new URL(buffer.uri, base)) : i === 0 ? container.bin : undefined;
    if (!bytes || bytes.byteLength < (buffer.byteLength as number) || (buffer.uri === undefined && bytes.byteLength > (buffer.byteLength as number) + 3)) throw new Error('GLTF_BUFFER_LENGTH');
    buffers.push(bytes);
  }
  const views = list(json.bufferViews, 'bufferViews'), images = new Map<number, TextureAsset>();
  // Only textures can reference core glTF images. Unused image entries still get
  // their metadata validated, but need no network fetch or bitmap allocation.
  const referencedImages = new Set<number>();
  for (const texture of list(json.textures, 'textures')) {
    if (typeof texture.source === 'number' && Number.isSafeInteger(texture.source) && texture.source >= 0) referencedImages.add(texture.source);
  }
  let decodedImageBytes = 0;
  for (const [i, image] of list(json.images, 'images').entries()) {
    const hasUri = image.uri !== undefined, hasView = image.bufferView !== undefined;
    if (hasUri === hasView || (hasUri && (typeof image.uri !== 'string' || image.uri.length === 0))) throw new Error('GLTF_IMAGE_SOURCE');
    if (image.mimeType !== undefined && image.mimeType !== 'image/png' && image.mimeType !== 'image/jpeg') throw new Error('GLTF_IMAGE_FORMAT');
    if (hasView && image.mimeType === undefined) throw new Error('GLTF_IMAGE_FORMAT: bufferView requires PNG or JPEG mimeType');
    let imageView: { buffer: ArrayBuffer; offset: number; length: number } | undefined;
    if (hasView) {
      const viewIndex = image.bufferView;
      if (typeof viewIndex !== 'number' || !Number.isSafeInteger(viewIndex) || viewIndex < 0 || viewIndex >= views.length) throw new Error('GLTF_IMAGE_BUFFER_VIEW');
      const view = views[viewIndex], bufferIndex = view.buffer;
      if (typeof bufferIndex !== 'number' || !Number.isSafeInteger(bufferIndex) || bufferIndex < 0 || bufferIndex >= buffers.length) throw new Error('GLTF_IMAGE_BUFFER_VIEW');
      const buffer = buffers[bufferIndex], declaredLength = bufferDefinitions[bufferIndex].byteLength as number;
      const offset = view.byteOffset ?? 0, length = view.byteLength;
      if (typeof offset !== 'number' || typeof length !== 'number' || !Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
        || offset < 0 || length < 1 || !Number.isSafeInteger(offset + length)
        || offset + length > declaredLength || offset + length > buffer.byteLength) throw new Error('GLTF_IMAGE_BOUNDS');
      imageView = { buffer, offset, length };
    }
    if (!referencedImages.has(i)) continue;
    if (maxDecodedImageBytes - decodedImageBytes < 4) throw new Error('GLTF_DECODED_IMAGE_BUDGET');
    const bytes = hasUri ? await read(new URL(image.uri as string, base))
      : imageView!.buffer.slice(imageView!.offset, imageView!.offset + imageView!.length);
    options.signal?.throwIfAborted();
    const decoded = await decodeTexture(new Blob([bytes], { type: (image.mimeType as string | undefined) ?? '' }), 'srgb', maxDecodedImageBytes - decodedImageBytes);
    decodedImageBytes += decoded.pixels.byteLength;
    images.set(i, decoded);
  }
  options.signal?.throwIfAborted();
  return importGltf(json, buffers, { ...options, images });
}
