import { decodeVerifiedGeometryPage, geometryPageTransferBuffers } from './geometry-page-decode';
import type { GeometryDecodePage } from './geometry-decode-worker-pool';

// This module runs only in a dedicated module Worker. DOM types are kept out of
// the public decoder contract; no GPU or renderer state crosses this boundary.
const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<{ type: string; id: number; page: GeometryDecodePage; bytes: Uint8Array<ArrayBuffer> }>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};
let busy = false;
scope.onmessage = async ({ data }) => {
  if (busy || data.type !== 'decode' || !Number.isSafeInteger(data.id)) {
    scope.postMessage({ type: 'failed', id: data.id, error: 'GEOMETRY_DECODE_PROTOCOL' }); return;
  }
  busy = true;
  try {
    const page = await decodeVerifiedGeometryPage(data.page, data.bytes);
    scope.postMessage({ type: 'decoded', id: data.id, page }, geometryPageTransferBuffers(page));
  } catch (error) {
    scope.postMessage({ type: 'failed', id: data.id, error: error instanceof Error ? error.message : String(error) });
  } finally { busy = false; }
};
