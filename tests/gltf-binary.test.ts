import { describe, expect, it } from 'vitest';
import { parseGlb, readAccessor, type GltfBinaryDocument } from '../src/gltf-binary';

function glb(chunks: { type: number; bytes: Uint8Array }[]): ArrayBuffer {
  const length = 12 + chunks.reduce((size, chunk) => size + 8 + chunk.bytes.length, 0);
  const bytes = new Uint8Array(length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, length, true);
  let offset = 12;
  for (const chunk of chunks) {
    view.setUint32(offset, chunk.bytes.length, true);
    view.setUint32(offset + 4, chunk.type, true);
    bytes.set(chunk.bytes, offset + 8);
    offset += 8 + chunk.bytes.length;
  }
  return bytes.buffer;
}

const json = new TextEncoder().encode('{"asset":{"version":"2.0"}} ');
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

describe('parseGlb', () => {
  it('reads JSON and a single BIN chunk, skipping later unknown chunks', () => {
    const binary = new Uint8Array([1, 2, 3, 0]);
    const parsed = parseGlb(glb([
      { type: JSON_CHUNK, bytes: json },
      { type: BIN_CHUNK, bytes: binary },
      { type: 0x12345678, bytes: new Uint8Array(4) },
    ]));
    expect(parsed.json).toEqual({ asset: { version: '2.0' } });
    expect(new Uint8Array(parsed.bin!)).toEqual(binary);
  });

  it('rejects bad headers, chunk order, alignment, and bounds', () => {
    const valid = glb([{ type: JSON_CHUNK, bytes: json }]);
    const badMagic = valid.slice(0);
    new DataView(badMagic).setUint32(0, 0, true);
    expect(() => parseGlb(badMagic)).toThrow(/magic/);
    const badLength = valid.slice(0);
    new DataView(badLength).setUint32(8, 24, true);
    expect(() => parseGlb(badLength)).toThrow(/declared length/);
    expect(() => parseGlb(glb([{ type: BIN_CHUNK, bytes: new Uint8Array(4) }]))).toThrow(/JSON chunk/);
    expect(() => parseGlb(glb([
      { type: JSON_CHUNK, bytes: json },
      { type: BIN_CHUNK, bytes: new Uint8Array(4) },
      { type: BIN_CHUNK, bytes: new Uint8Array(4) },
    ]))).toThrow(/BIN chunk/);
    const badChunkLength = valid.slice(0);
    new DataView(badChunkLength).setUint32(12, 5, true);
    expect(() => parseGlb(badChunkLength)).toThrow(/Unaligned/);
    new DataView(badChunkLength).setUint32(12, 100, true);
    expect(() => parseGlb(badChunkLength)).toThrow(/out of bounds/);
  });
});

describe('readAccessor', () => {
  it('decodes interleaved normalized weights and exact unsigned indices', () => {
    const buffer = new Uint8Array(16);
    buffer.set([0, 128, 255, 64, 9, 0, 0, 0, 255, 0, 0, 0, 10, 0, 0, 0]);
    const doc: GltfBinaryDocument = {
      buffers: [{ byteLength: 16 }],
      bufferViews: [{ buffer: 0, byteLength: 16, byteStride: 8 }],
      accessors: [
        { bufferView: 0, componentType: 5121, count: 2, type: 'VEC4', normalized: true },
        { bufferView: 0, byteOffset: 4, componentType: 5125, count: 2, type: 'SCALAR' },
      ],
    };
    const weights = readAccessor(doc, [buffer.buffer], 0);
    expect(weights.values).toBeInstanceOf(Float32Array);
    expect(Array.from(weights.values)).toEqual([0, Math.fround(128 / 255), 1, Math.fround(64 / 255), 1, 0, 0, 0]);
    const indices = readAccessor(doc, [buffer.buffer], 1);
    expect(indices.values).toBeInstanceOf(Uint32Array);
    expect(Array.from(indices.values)).toEqual([9, 10]);
  });

  it('applies sparse morph values over an implicit zero base', () => {
    const bytes = new Uint8Array(32);
    bytes.set([1, 3], 0);
    const view = new DataView(bytes.buffer);
    view.setFloat32(4, 1.5, true);
    view.setFloat32(8, -2, true);
    view.setFloat32(12, 3, true);
    view.setFloat32(16, -4, true);
    view.setFloat32(20, 5, true);
    view.setFloat32(24, 6, true);
    const doc: GltfBinaryDocument = {
      buffers: [{ byteLength: 32 }],
      bufferViews: [{ buffer: 0, byteLength: 2 }, { buffer: 0, byteOffset: 4, byteLength: 24 }],
      accessors: [{ componentType: 5126, count: 4, type: 'VEC3', sparse: {
        count: 2,
        indices: { bufferView: 0, componentType: 5121 },
        values: { bufferView: 1 },
      } }],
    };
    expect(Array.from(readAccessor(doc, [bytes.buffer], 0).values)).toEqual([
      0, 0, 0, 1.5, -2, 3, 0, 0, 0, -4, 5, 6,
    ]);
    bytes[1] = 1;
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/strictly increase/);
  });

  it('skips glTF matrix column padding', () => {
    const bytes = new Uint8Array([
      1, 2, 0, 0, 3, 4, 0, 0,
      5, 6, 0, 0, 7, 8, 0, 0,
    ]);
    const doc: GltfBinaryDocument = {
      bufferViews: [{ buffer: 0, byteLength: bytes.length }],
      accessors: [{ bufferView: 0, componentType: 5121, count: 2, type: 'MAT2' }],
    };
    expect(Array.from(readAccessor(doc, [bytes.buffer], 0).values)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);

    const matrix = new ArrayBuffer(24);
    const matrixView = new DataView(matrix);
    for (let column = 0; column < 3; column++) {
      for (let row = 0; row < 3; row++) {
        matrixView.setUint16(column * 8 + row * 2, column * 3 + row + 1, true);
      }
    }
    doc.bufferViews![0].byteLength = 22; // Final trailing column padding may be absent.
    doc.accessors![0] = { bufferView: 0, componentType: 5123, count: 1, type: 'MAT3' };
    expect(Array.from(readAccessor(doc, [matrix], 0).values)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('maps signed normalized minimum values to exactly -1', () => {
    const bytes = new Int8Array([-128, 127, -64, 0]);
    const doc: GltfBinaryDocument = {
      bufferViews: [{ buffer: 0, byteLength: 4 }],
      accessors: [{ bufferView: 0, componentType: 5120, count: 1, type: 'VEC4', normalized: true }],
    };
    expect(Array.from(readAccessor(doc, [bytes.buffer], 0).values)).toEqual([-1, 1, Math.fround(-64 / 127), 0]);
  });

  it('accepts a 2-byte aligned ushort bufferView and rejects a misaligned absolute offset', () => {
    const buffer = new ArrayBuffer(12);
    const view = new DataView(buffer);
    view.setUint16(2, 7, true);
    view.setUint16(4, 11, true);
    const doc: GltfBinaryDocument = {
      buffers: [{ byteLength: 12 }],
      bufferViews: [{ buffer: 0, byteOffset: 2, byteLength: 4 }],
      accessors: [{ bufferView: 0, componentType: 5123, count: 2, type: 'SCALAR' }],
    };
    expect(Array.from(readAccessor(doc, [buffer], 0).values)).toEqual([7, 11]);
    doc.bufferViews![0].byteOffset = 1;
    expect(() => readAccessor(doc, [buffer], 0)).toThrow(/misaligned/);
  });

  it('rejects unsafe and out-of-range accessors before allocation or reading', () => {
    const bytes = new Uint8Array(8);
    const doc: GltfBinaryDocument = {
      buffers: [{ byteLength: 8 }],
      bufferViews: [{ buffer: 0, byteLength: 8 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' }],
    };
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/out of bounds/);
    doc.accessors![0] = { bufferView: 0, componentType: 5121, count: 10, type: 'SCALAR' };
    expect(() => readAccessor(doc, [bytes.buffer], 0, { maxElements: 4 })).toThrow(/element limit/);
    doc.accessors![0] = { bufferView: 0, componentType: 5126, count: 1, type: 'SCALAR', normalized: true };
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/Normalized FLOAT/);
    doc.accessors![0] = { bufferView: 0, componentType: 5126, count: 1, type: 'SCALAR' };
    new DataView(bytes.buffer).setFloat32(0, Infinity, true);
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/non-finite/);
    doc.bufferViews![0].byteStride = 2;
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/byteStride/);
    doc.bufferViews![0].byteStride = undefined;
    doc.buffers![0].byteLength = 9;
    expect(() => readAccessor(doc, [bytes.buffer], 0)).toThrow(/shorter than declared/);
  });
});
