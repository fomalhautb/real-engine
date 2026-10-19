import type { Bounds3 } from './deformation';

/** Compact, immutable build metadata. No full source vertex/index arrays are required. */
export interface GeometryStreamCluster {
  readonly level: number;
  readonly group: number;
  readonly refinedGroup: number | null;
  readonly vertexCount: number;
  readonly triangleCount: number;
  readonly bounds: Bounds3;
  readonly lodBounds: Bounds3;
  readonly error: number;
}
export interface GeometryStreamGroup {
  readonly level: number;
  readonly children: readonly number[];
  readonly parents: readonly number[];
  readonly bounds: Bounds3;
  readonly error: number;
  readonly terminal: boolean;
}
export interface GeometryStreamPage {
  readonly id: number;
  readonly clusterIds: readonly number[];
  readonly encodedBytes: number;
  readonly decodedBytes: number;
  readonly sha256: string;
}
export interface GeometryStreamDirectory {
  readonly version: 1;
  readonly primitiveKey: string;
  readonly sourceVertexCount: number;
  readonly sourceTriangleCount: number;
  readonly vertexStride: number;
  /** Present only when pages carry linear RGBA. Omitted in legacy directories. */
  readonly hasColors?: true;
  readonly clusters: readonly GeometryStreamCluster[];
  readonly groups: readonly GeometryStreamGroup[];
  readonly roots: readonly number[];
  readonly pages: readonly GeometryStreamPage[];
  readonly clusterPages: readonly number[];
  readonly rootPages: readonly number[];
}

/** Providers must honor abort and return at most descriptor.encodedBytes bytes.
 * Directory hash/length and decoded topology are validated by the cache afterward. */
export type GeometryPageProvider = (page: GeometryStreamPage, signal: AbortSignal) => Promise<Uint8Array>;
