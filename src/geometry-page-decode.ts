import { decodeGeometryPage, type DecodedGeometryPage } from './geometry-pages';
import type { GeometryStreamPage } from './geometry-stream-types';

/** Verify and decode an owned snapshot. The caller must not mutate or detach it
 * until this operation settles; the cache snapshots external provider buffers. */
export async function decodeVerifiedGeometryPage(
  page: Pick<GeometryStreamPage, 'id' | 'encodedBytes' | 'decodedBytes' | 'sha256'>,
  bytes: Uint8Array<ArrayBuffer>, signal?: AbortSignal,
): Promise<DecodedGeometryPage> {
  signal?.throwIfAborted();
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== page.encodedBytes) {
    throw new Error('GEOMETRY_STREAM_ENCODED_LENGTH: ' + page.id);
  }
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  signal?.throwIfAborted();
  const actual = Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('');
  if (actual !== page.sha256) throw new Error('GEOMETRY_STREAM_HASH: ' + page.id);
  const decoded = await decodeGeometryPage(bytes, {
    maxPageBytes: page.decodedBytes, maxEncodedBytes: page.encodedBytes,
  });
  signal?.throwIfAborted();
  return decoded;
}

/** Return unique transferable cluster buffers; shared buffers stay with their owner. */
export function geometryPageTransferBuffers(decoded: DecodedGeometryPage): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>();
  for (const cluster of decoded.clusters) {
    for (const array of [cluster.sourceVertexIds, cluster.rest, cluster.uvs, cluster.tangents, cluster.localTriangles]) {
      if (array?.buffer instanceof ArrayBuffer) buffers.add(array.buffer);
    }
  }
  return [...buffers];
}
