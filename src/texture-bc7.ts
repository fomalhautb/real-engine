/**
 * Deterministic CPU BC7 (BPTC UNORM) block codec for offline asset builds.
 *
 * Self-contained on purpose (no imports, erasable TypeScript only) so Node
 * worker threads can load it directly and browsers/tests can reuse the exact
 * decoder. Encoding is a pure function of the 4x4 RGBA8 input and options: no
 * randomness, clocks, threads or platform-dependent math (only +,-,*,/,sqrt and
 * integer operations), so splitting an image across workers cannot change the
 * output. Values are encoded as stored bytes; for sRGB textures the error is
 * therefore measured on sRGB-encoded bytes, as GPU sampling decodes after BC7.
 */

const W2 = [0, 21, 43, 64];
const W3 = [0, 9, 18, 27, 37, 46, 55, 64];
const W4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
const WEIGHTS: readonly (readonly number[])[] = [[], [], W2, W3, W4];

interface Bc7Mode {
  readonly subsets: number; readonly partitionBits: number; readonly rotationBits: number;
  readonly indexSelectionBits: number; readonly colorBits: number; readonly alphaBits: number;
  readonly endpointPBits: number; readonly sharedPBits: number; readonly indexBits: number; readonly indexBits2: number;
}
const MODES: readonly Bc7Mode[] = [
  { subsets: 3, partitionBits: 4, rotationBits: 0, indexSelectionBits: 0, colorBits: 4, alphaBits: 0, endpointPBits: 1, sharedPBits: 0, indexBits: 3, indexBits2: 0 },
  { subsets: 2, partitionBits: 6, rotationBits: 0, indexSelectionBits: 0, colorBits: 6, alphaBits: 0, endpointPBits: 0, sharedPBits: 1, indexBits: 3, indexBits2: 0 },
  { subsets: 3, partitionBits: 6, rotationBits: 0, indexSelectionBits: 0, colorBits: 5, alphaBits: 0, endpointPBits: 0, sharedPBits: 0, indexBits: 2, indexBits2: 0 },
  { subsets: 2, partitionBits: 6, rotationBits: 0, indexSelectionBits: 0, colorBits: 7, alphaBits: 0, endpointPBits: 1, sharedPBits: 0, indexBits: 2, indexBits2: 0 },
  { subsets: 1, partitionBits: 0, rotationBits: 2, indexSelectionBits: 1, colorBits: 5, alphaBits: 6, endpointPBits: 0, sharedPBits: 0, indexBits: 2, indexBits2: 3 },
  { subsets: 1, partitionBits: 0, rotationBits: 2, indexSelectionBits: 0, colorBits: 7, alphaBits: 8, endpointPBits: 0, sharedPBits: 0, indexBits: 2, indexBits2: 2 },
  { subsets: 1, partitionBits: 0, rotationBits: 0, indexSelectionBits: 0, colorBits: 7, alphaBits: 7, endpointPBits: 1, sharedPBits: 0, indexBits: 4, indexBits2: 0 },
  { subsets: 2, partitionBits: 6, rotationBits: 0, indexSelectionBits: 0, colorBits: 5, alphaBits: 5, endpointPBits: 1, sharedPBits: 0, indexBits: 2, indexBits2: 0 },
];

// Two-subset partitions as 16-bit masks; bit i (pixel y*4+x) selects subset 1.
const MASKS2 = [
  0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
  0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
  0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
  0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
];
const PARTITIONS3 = [
  '0011001102212222', '0001001122112221', '0000200122112211', '0222002200110111', '0000000011221122', '0011001100220022', '0022002211111111', '0011001122112211',
  '0000000011112222', '0000111111112222', '0000111122222222', '0012001200120012', '0112011201120112', '0122012201220122', '0011011211221222', '0011200122002220',
  '0001001101121122', '0111001120012200', '0000112211221122', '0022002200221111', '0111011102220222', '0001000122212221', '0000001101220122', '0000110022102210',
  '0122012200110000', '0012001211222222', '0110122112210110', '0000011012211221', '0022110211020022', '0110011020022222', '0011012201220011', '0000200022112221',
  '0000000211221222', '0222002200120011', '0011001200220222', '0120012001200120', '0000111122220000', '0120120120120120', '0120201212010120', '0011220011220011',
  '0011112222000011', '0101010122222222', '0000000021212121', '0022112200221122', '0022001100220011', '0220122102201221', '0101222222220101', '0000212121212121',
  '0101010101012222', '0222011102220111', '0002111200021112', '0000211221122112', '0222011101110222', '0002111211120002', '0110011001102222', '0000000021122112',
  '0110011022222222', '0022001100110022', '0022112211220022', '0000000000002112', '0002000100020001', '0222122202221222', '0101222222222222', '0111201122012220',
];
const ANCHOR2 = [
  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
  15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A = [
  3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
  8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B = [
  15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
  15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/** subset[ns][partition * 16 + pixel] for ns = 1..3. */
const SUBSET: Uint8Array[] = [new Uint8Array(0), new Uint8Array(64 * 16), new Uint8Array(64 * 16), new Uint8Array(64 * 16)];
/** anchor[ns][partition * 3 + subset]. */
const ANCHOR: Uint8Array[] = [new Uint8Array(0), new Uint8Array(64 * 3), new Uint8Array(64 * 3), new Uint8Array(64 * 3)];
for (let p = 0; p < 64; p++) {
  for (let i = 0; i < 16; i++) {
    SUBSET[2][p * 16 + i] = (MASKS2[p] >> i) & 1;
    SUBSET[3][p * 16 + i] = PARTITIONS3[p].charCodeAt(i) - 48;
  }
  ANCHOR[2][p * 3 + 1] = ANCHOR2[p];
  ANCHOR[3][p * 3 + 1] = ANCHOR3A[p];
  ANCHOR[3][p * 3 + 2] = ANCHOR3B[p];
}

/** Exposed for table self-consistency tests. */
export const BC7_TABLES = Object.freeze({ subsets: SUBSET, anchors: ANCHOR, weights: WEIGHTS });

function unquantize(value: number, bits: number): number {
  if (bits >= 8) return value;
  const shifted = value << (8 - bits);
  return shifted | (shifted >> bits);
}

// ---------------------------------------------------------------- decoding

const decEndpoints = new Int32Array(24), decPbits = new Int32Array(6);
const decPrimary = new Uint8Array(16), decSecondary = new Uint8Array(16);
let decSource: Uint8Array = new Uint8Array(16), decOffset = 0, decBit = 0;
function read(count: number): number {
  let value = 0;
  for (let i = 0; i < count; i++, decBit++) value |= ((decSource[decOffset + (decBit >> 3)] >> (decBit & 7)) & 1) << i;
  return value;
}

/** Decodes one 16-byte block into 16 RGBA8 pixels (row-major, 64 bytes). */
export function decodeBc7Block(source: Uint8Array, offset: number, out: Uint8Array, outOffset = 0): void {
  decSource = source; decOffset = offset; decBit = 0;
  let mode = 0;
  while (mode < 8 && read(1) === 0) mode++;
  if (mode === 8) { out.fill(0, outOffset, outOffset + 64); return; }
  const info = MODES[mode];
  const ns = info.subsets;
  const partition = read(info.partitionBits);
  const rotation = read(info.rotationBits);
  const selection = read(info.indexSelectionBits);
  const endpoints = decEndpoints; // [endpoint * 4 + channel]
  endpoints.fill(0);
  for (let c = 0; c < 3; c++) for (let e = 0; e < 2 * ns; e++) endpoints[e * 4 + c] = read(info.colorBits);
  if (info.alphaBits) for (let e = 0; e < 2 * ns; e++) endpoints[e * 4 + 3] = read(info.alphaBits);
  const pbits = decPbits;
  pbits.fill(0);
  if (info.endpointPBits) for (let e = 0; e < 2 * ns; e++) pbits[e] = read(1);
  if (info.sharedPBits) for (let s = 0; s < ns; s++) pbits[2 * s] = pbits[2 * s + 1] = read(1);
  const hasP = info.endpointPBits + info.sharedPBits > 0 ? 1 : 0;
  for (let e = 0; e < 2 * ns; e++) {
    for (let c = 0; c < 4; c++) {
      if (c === 3 && !info.alphaBits) { endpoints[e * 4 + 3] = 255; continue; }
      const bits = (c < 3 ? info.colorBits : info.alphaBits) + hasP;
      const raw = hasP ? (endpoints[e * 4 + c] << 1) | pbits[e] : endpoints[e * 4 + c];
      endpoints[e * 4 + c] = unquantize(raw, bits);
    }
  }
  const subsets = ns > 1 ? SUBSET[ns] : null;
  const primary = decPrimary, secondary = decSecondary;
  for (let i = 0; i < 16; i++) {
    const s = subsets ? subsets[partition * 16 + i] : 0;
    const anchor = s === 0 ? i === 0 : ANCHOR[ns][partition * 3 + s] === i;
    primary[i] = read(info.indexBits - (anchor ? 1 : 0));
  }
  if (info.indexBits2) for (let i = 0; i < 16; i++) secondary[i] = read(info.indexBits2 - (i === 0 ? 1 : 0));
  const colorBits = info.indexBits2 && selection ? info.indexBits2 : info.indexBits;
  const alphaBits = info.indexBits2 ? (selection ? info.indexBits : info.indexBits2) : info.indexBits;
  for (let i = 0; i < 16; i++) {
    const s = subsets ? subsets[partition * 16 + i] : 0;
    const colorIndex = info.indexBits2 && selection ? secondary[i] : primary[i];
    const alphaIndex = info.indexBits2 ? (selection ? primary[i] : secondary[i]) : primary[i];
    const wc = WEIGHTS[colorBits][colorIndex], wa = WEIGHTS[alphaBits][alphaIndex];
    const o = outOffset + i * 4;
    for (let c = 0; c < 4; c++) {
      const w = c < 3 ? wc : wa;
      out[o + c] = ((64 - w) * endpoints[(2 * s) * 4 + c] + w * endpoints[(2 * s + 1) * 4 + c] + 32) >> 6;
    }
    if (rotation) { const t = out[o + 3]; out[o + 3] = out[o + rotation - 1]; out[o + rotation - 1] = t; }
  }
}

/** Decodes a tightly packed BC7 level into RGBA8 pixels of the given size. */
export function decodeBc7Image(blocks: Uint8Array, width: number, height: number): Uint8Array {
  const bw = Math.ceil(width / 4), bh = Math.ceil(height / 4);
  if (blocks.byteLength !== bw * bh * 16) throw new RangeError('BC7: block byte length does not match dimensions');
  const out = new Uint8Array(width * height * 4), tile = new Uint8Array(64);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      decodeBc7Block(blocks, (by * bw + bx) * 16, tile);
      for (let y = 0; y < 4 && by * 4 + y < height; y++) {
        for (let x = 0; x < 4 && bx * 4 + x < width; x++) {
          out.set(tile.subarray((y * 4 + x) * 4, (y * 4 + x) * 4 + 4), ((by * 4 + y) * width + bx * 4 + x) * 4);
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- encoding

export interface Bc7EncodeOptions {
  /** 0: mode 6 only; 1: adds ranked two-subset and rotated separate-alpha modes; 2: all eight modes, wider search. Default 1. */
  readonly quality?: number;
  /** Squared-error channel weights for RGBA; default 1,1,1,1. A zero alpha weight marks alpha as unused (any decoded alpha is acceptable, opaque blocks still decode 255). */
  readonly channelWeights?: readonly [number, number, number, number];
  /** Stop trying further modes once the block's weighted squared error is at or below this value (default 0). */
  readonly earlyOutError?: number;
}

interface Candidate {
  mode: number; partition: number; rotation: number; selection: number; error: number;
  endpoints: Int32Array; // quantized codes [endpoint * 4 + channel], without p-bits
  pbits: Int32Array; // per endpoint (shared bits duplicated)
  primary: Uint8Array; secondary: Uint8Array;
}
function candidate(): Candidate {
  return { mode: -1, partition: 0, rotation: 0, selection: 0, error: Infinity, endpoints: new Int32Array(24),
    pbits: new Int32Array(6), primary: new Uint8Array(16), secondary: new Uint8Array(16) };
}
function copyCandidate(to: Candidate, from: Candidate): void {
  to.mode = from.mode; to.partition = from.partition; to.rotation = from.rotation; to.selection = from.selection;
  to.error = from.error; to.endpoints.set(from.endpoints); to.pbits.set(from.pbits);
  to.primary.set(from.primary); to.secondary.set(from.secondary);
}

// Scratch state; every worker owns its own module instance (no sharing).
const block = new Int32Array(64); // working (possibly rotated) pixels
const original = new Int32Array(64);
const weight = new Float64Array(4), baseWeight = new Float64Array(4);
const best = candidate(), trial = candidate();
const list = new Uint8Array(16);
const fe = new Float64Array(8); // float endpoints e0[0..3], e1[0..3]
const qe = new Int32Array(8), qp = new Int32Array(2); // quantized codes and p-bits
const de = new Int32Array(8); // dequantized endpoint values
const idx = new Uint8Array(16), bestIdx = new Uint8Array(16), colorIdx = new Uint8Array(16);
const bestQe = new Int32Array(8), bestQp = new Int32Array(2);
const palette = new Int32Array(64);
const mean = new Float64Array(4), cov = new Float64Array(16), axis = new Float64Array(4), next = new Float64Array(4);
const lsX = new Float64Array(4), lsY = new Float64Array(4);
const sums = new Float64Array(3 * 16); // per subset: count, 4 sums, 10 products
const estimates2 = new Float64Array(64), estimates3 = new Float64Array(64);
const order2 = new Int32Array(64), order3 = new Int32Array(64);
let estimated2 = false, estimated3 = false;
/** Opaque blocks penalize any decoded alpha other than 255 so alpha stays exact. */
const OPAQUE_ALPHA_WEIGHT = 1e6;

/** Quantizes a channel value; bits is the total precision including the p-bit when pbit >= 0. */
function quantizeChannel(value: number, bits: number, pbit: number): number {
  if (pbit < 0) {
    const max = (1 << bits) - 1;
    let q = Math.floor(value * max / 255 + 0.5);
    if (q < 0) q = 0; else if (q > max) q = max;
    let bestQ = q, bestError = Math.abs(unquantize(q, bits) - value);
    if (q > 0) { const error = Math.abs(unquantize(q - 1, bits) - value); if (error < bestError) { bestError = error; bestQ = q - 1; } }
    if (q < max) { const error = Math.abs(unquantize(q + 1, bits) - value); if (error < bestError) { bestQ = q + 1; } }
    return bestQ;
  }
  const max = (1 << (bits - 1)) - 1;
  let q = Math.floor(((value * ((1 << bits) - 1) / 255) - pbit) / 2 + 0.5);
  if (q < 0) q = 0; else if (q > max) q = max;
  let bestQ = q, bestError = Math.abs(unquantize((q << 1) | pbit, bits) - value);
  if (q > 0) { const error = Math.abs(unquantize(((q - 1) << 1) | pbit, bits) - value); if (error < bestError) { bestError = error; bestQ = q - 1; } }
  if (q < max) { const error = Math.abs(unquantize(((q + 1) << 1) | pbit, bits) - value); if (error < bestError) { bestQ = q + 1; } }
  return bestQ;
}

/** Nearest palette indices for pixels in list over channel range; returns weighted squared error. */
function assign(count: number, ch0: number, chN: number, levels: number, out: Uint8Array): number {
  const w = levels === 4 ? W2 : levels === 8 ? W3 : W4;
  for (let k = 0; k < levels; k++) {
    const wk = w[k], iw = 64 - wk;
    for (let c = ch0; c < ch0 + chN; c++) palette[k * 4 + c] = (iw * de[c] + wk * de[4 + c] + 32) >> 6;
  }
  const w0 = weight[0], w1 = weight[1], w2 = weight[2], w3 = weight[3];
  let total = 0;
  for (let n = 0; n < count; n++) {
    const p = list[n] * 4;
    let bestError = Infinity, bestK = 0;
    if (chN === 1) {
      const a = block[p + ch0], wa = weight[ch0];
      for (let k = 0; k < levels; k++) {
        const d = palette[k * 4 + ch0] - a, error = wa * d * d;
        if (error < bestError) { bestError = error; bestK = k; }
      }
    } else if (chN === 3) {
      const r = block[p], g = block[p + 1], b = block[p + 2];
      for (let k = 0; k < levels; k++) {
        const q = k * 4, dr = palette[q] - r, dg = palette[q + 1] - g, db = palette[q + 2] - b;
        const error = w0 * dr * dr + w1 * dg * dg + w2 * db * db;
        if (error < bestError) { bestError = error; bestK = k; }
      }
    } else {
      const r = block[p], g = block[p + 1], b = block[p + 2], a = block[p + 3];
      for (let k = 0; k < levels; k++) {
        const q = k * 4, dr = palette[q] - r, dg = palette[q + 1] - g, db = palette[q + 2] - b, da = palette[q + 3] - a;
        const error = w0 * dr * dr + w1 * dg * dg + w2 * db * db + w3 * da * da;
        if (error < bestError) { bestError = error; bestK = k; }
      }
    }
    out[list[n]] = bestK;
    total += bestError;
  }
  return total;
}

/** Principal-axis endpoints over channels [ch0, ch0+chN) for pixels in list. */
function principalEndpoints(count: number, ch0: number, chN: number): void {
  mean.fill(0); cov.fill(0);
  for (let n = 0; n < count; n++) for (let c = 0; c < chN; c++) mean[c] += block[list[n] * 4 + ch0 + c];
  for (let c = 0; c < chN; c++) mean[c] /= count;
  for (let n = 0; n < count; n++) {
    const p = list[n] * 4 + ch0;
    for (let a = 0; a < chN; a++) {
      const da = block[p + a] - mean[a];
      for (let b = a; b < chN; b++) cov[a * 4 + b] += da * (block[p + b] - mean[b]);
    }
  }
  for (let a = 0; a < chN; a++) for (let b = 0; b < a; b++) cov[a * 4 + b] = cov[b * 4 + a];
  axis.fill(0);
  let largest = -1, start = 0;
  for (let c = 0; c < chN; c++) if (cov[c * 4 + c] > largest) { largest = cov[c * 4 + c]; start = c; }
  axis[start] = 1;
  if (largest > 0) {
    for (let iteration = 0; iteration < 8; iteration++) {
      let length = 0;
      for (let a = 0; a < chN; a++) {
        let v = 0;
        for (let b = 0; b < chN; b++) v += cov[a * 4 + b] * axis[b];
        next[a] = v; length += v * v;
      }
      length = Math.sqrt(length);
      if (!(length > 1e-12)) break;
      for (let c = 0; c < chN; c++) axis[c] = next[c] / length;
    }
  }
  let lo = Infinity, hi = -Infinity;
  for (let n = 0; n < count; n++) {
    const p = list[n] * 4 + ch0;
    let t = 0;
    for (let c = 0; c < chN; c++) t += (block[p + c] - mean[c]) * axis[c];
    if (t < lo) lo = t;
    if (t > hi) hi = t;
  }
  if (!(hi >= lo)) { lo = 0; hi = 0; }
  for (let c = 0; c < chN; c++) {
    fe[ch0 + c] = Math.min(255, Math.max(0, mean[c] + lo * axis[c]));
    fe[4 + ch0 + c] = Math.min(255, Math.max(0, mean[c] + hi * axis[c]));
  }
}

/** Least-squares endpoint refit from indices; false if the system is singular. */
function leastSquares(count: number, ch0: number, chN: number, levels: number, indices: Uint8Array): boolean {
  const w = levels === 4 ? W2 : levels === 8 ? W3 : W4;
  let aa = 0, ab = 0, bb = 0;
  lsX.fill(0); lsY.fill(0);
  for (let n = 0; n < count; n++) {
    const t = w[indices[list[n]]] / 64, s = 1 - t;
    aa += s * s; ab += s * t; bb += t * t;
    for (let c = 0; c < chN; c++) { const v = block[list[n] * 4 + ch0 + c]; lsX[c] += s * v; lsY[c] += t * v; }
  }
  const det = aa * bb - ab * ab;
  if (!(Math.abs(det) > 1e-9)) return false;
  for (let c = 0; c < chN; c++) {
    fe[ch0 + c] = Math.min(255, Math.max(0, (bb * lsX[c] - ab * lsY[c]) / det));
    fe[4 + ch0 + c] = Math.min(255, Math.max(0, (aa * lsY[c] - ab * lsX[c]) / det));
  }
  return true;
}

/** Quantizes fe into qe/qp/de. pbitMode: 0 none, 1 per endpoint (p0/p1), 2 shared (p0). */
function quantizeEndpoints(ch0: number, chN: number, bitsColor: number, bitsAlpha: number, pbitMode: number, p0: number, p1: number): void {
  for (let e = 0; e < 2; e++) {
    const p = pbitMode === 0 ? -1 : e === 0 || pbitMode === 2 ? p0 : p1;
    qp[e] = p < 0 ? 0 : p;
    for (let c = ch0; c < ch0 + chN; c++) {
      const bits = (c < 3 ? bitsColor : bitsAlpha) + (p < 0 ? 0 : 1);
      const code = quantizeChannel(fe[e * 4 + c], bits, p);
      qe[e * 4 + c] = code;
      de[e * 4 + c] = unquantize(p < 0 ? code : (code << 1) | p, bits);
    }
  }
}

/** Best p-bit for one endpoint by endpoint quantization error (multi-subset heuristic). */
function bestEndpointPBit(e: number, ch0: number, chN: number, bitsColor: number, bitsAlpha: number): number {
  let bestError = Infinity, bestP = 0;
  for (let p = 0; p < 2; p++) {
    let error = 0;
    for (let c = ch0; c < ch0 + chN; c++) {
      const bits = (c < 3 ? bitsColor : bitsAlpha) + 1;
      const v = unquantize((quantizeChannel(fe[e * 4 + c], bits, p) << 1) | p, bits) - fe[e * 4 + c];
      error += weight[c] * v * v;
    }
    if (error < bestError) { bestError = error; bestP = p; }
  }
  return bestP;
}

/** Fits one subset (pixels in list); best codes/p-bits in bestQe/bestQp, indices in bestIdx. */
function fitSubset(count: number, ch0: number, chN: number, levels: number, bitsColor: number, bitsAlpha: number,
  pbitMode: number, exhaustivePBits: boolean, iterations: number): number {
  principalEndpoints(count, ch0, chN);
  let bestError = Infinity;
  const pbitTrials = pbitMode === 0 ? 1 : pbitMode === 2 ? 2 : exhaustivePBits ? 4 : 1;
  for (let iteration = 0; iteration <= iterations; iteration++) {
    let improved = false;
    for (let trialP = 0; trialP < pbitTrials; trialP++) {
      let p0 = 0, p1 = 0;
      if (pbitMode === 1) {
        if (exhaustivePBits) { p0 = trialP & 1; p1 = trialP >> 1; }
        else { p0 = bestEndpointPBit(0, ch0, chN, bitsColor, bitsAlpha); p1 = bestEndpointPBit(1, ch0, chN, bitsColor, bitsAlpha); }
      } else if (pbitMode === 2) p0 = trialP;
      quantizeEndpoints(ch0, chN, bitsColor, bitsAlpha, pbitMode, p0, p1);
      const error = assign(count, ch0, chN, levels, idx);
      if (error < bestError) {
        bestError = error; improved = true;
        bestQe.set(qe); bestQp.set(qp);
        for (let n = 0; n < count; n++) bestIdx[list[n]] = idx[list[n]];
      }
    }
    if (iteration === iterations || bestError === 0 || (!improved && iteration > 0)) break;
    if (!leastSquares(count, ch0, chN, levels, bestIdx)) break;
  }
  return bestError;
}

/** Sum over subsets of squared distance to each subset's best-fit RGB(A) line. */
function partitionEstimate(ns: number, partition: number, chN: number): number {
  const table = SUBSET[ns];
  sums.fill(0);
  for (let i = 0; i < 16; i++) {
    const o = table[partition * 16 + i] * 16, p = i * 4;
    sums[o]++;
    let q = o + 5;
    for (let a = 0; a < chN; a++) {
      const va = block[p + a];
      sums[o + 1 + a] += va;
      for (let b = a; b < chN; b++) sums[q++] += va * block[p + b];
    }
  }
  let total = 0;
  for (let s = 0; s < ns; s++) {
    const o = s * 16, count = sums[o];
    if (count < 2) continue;
    let q = o + 5, trace = 0;
    for (let a = 0; a < chN; a++) {
      for (let b = a; b < chN; b++) {
        const v = (sums[q++] - sums[o + 1 + a] * sums[o + 1 + b] / count) * Math.sqrt(weight[a] * weight[b]);
        cov[a * 4 + b] = v; cov[b * 4 + a] = v;
        if (a === b) trace += v;
      }
    }
    for (let c = 0; c < chN; c++) axis[c] = 1;
    let lambda = 0;
    for (let iteration = 0; iteration < 6; iteration++) {
      let length = 0;
      for (let a = 0; a < chN; a++) {
        let v = 0;
        for (let b = 0; b < chN; b++) v += cov[a * 4 + b] * axis[b];
        next[a] = v; length += v * v;
      }
      length = Math.sqrt(length);
      if (!(length > 1e-12)) { lambda = 0; break; }
      lambda = length;
      for (let c = 0; c < chN; c++) axis[c] = next[c] / length;
    }
    total += Math.max(0, trace - lambda);
  }
  return total;
}

function rankPartitions(ns: number, chN: number, estimates: Float64Array, order: Int32Array): void {
  for (let p = 0; p < 64; p++) { estimates[p] = partitionEstimate(ns, p, chN); order[p] = p; }
  // Insertion sort: ascending estimate, ties by partition index (deterministic).
  for (let i = 1; i < 64; i++) {
    const value = order[i];
    let j = i - 1;
    while (j >= 0 && (estimates[order[j]] > estimates[value] || (estimates[order[j]] === estimates[value] && order[j] > value))) { order[j + 1] = order[j]; j--; }
    order[j + 1] = value;
  }
}

/** Single-subset RGBA mode 6 with exhaustive endpoint p-bits. */
function tryMode6(): void {
  trial.mode = 6; trial.partition = 0; trial.rotation = 0; trial.selection = 0;
  for (let i = 0; i < 16; i++) list[i] = i;
  const error = fitSubset(16, 0, 4, 16, 7, 7, 1, true, 2);
  trial.endpoints.fill(0); trial.pbits.fill(0);
  for (let e = 0; e < 8; e++) trial.endpoints[e] = bestQe[e];
  trial.pbits[0] = bestQp[0]; trial.pbits[1] = bestQp[1];
  trial.primary.set(bestIdx);
  trial.error = error;
  if (trial.error < best.error) copyCandidate(best, trial);
}

/** Separate-alpha modes 4/5 with a channel rotation and index selection. */
function tryMode45(mode: number, rotation: number, selection: number): void {
  const info = MODES[mode];
  for (let i = 0; i < 64; i++) block[i] = original[i];
  if (rotation) {
    const r = rotation - 1;
    for (let i = 0; i < 16; i++) { const t = block[i * 4 + 3]; block[i * 4 + 3] = block[i * 4 + r]; block[i * 4 + r] = t; }
    const t = weight[3]; weight[3] = weight[r]; weight[r] = t;
  }
  for (let i = 0; i < 16; i++) list[i] = i;
  const colorLevels = 1 << (selection ? info.indexBits2 : info.indexBits);
  const alphaLevels = 1 << (selection ? info.indexBits : info.indexBits2);
  trial.mode = mode; trial.partition = 0; trial.rotation = rotation; trial.selection = selection;
  trial.endpoints.fill(0); trial.pbits.fill(0);
  let error = fitSubset(16, 0, 3, colorLevels, info.colorBits, info.alphaBits, 0, false, 2);
  for (let e = 0; e < 2; e++) for (let c = 0; c < 3; c++) trial.endpoints[e * 4 + c] = bestQe[e * 4 + c];
  colorIdx.set(bestIdx);
  error += fitSubset(16, 3, 1, alphaLevels, info.colorBits, info.alphaBits, 0, false, 2);
  trial.endpoints[3] = bestQe[3]; trial.endpoints[7] = bestQe[7];
  // The bitstream's primary set holds the 2-bit indices; mode 4 selection=1 stores color in the 3-bit set.
  if (selection) { trial.secondary.set(colorIdx); trial.primary.set(bestIdx); }
  else { trial.primary.set(colorIdx); trial.secondary.set(bestIdx); }
  for (let c = 0; c < 4; c++) weight[c] = baseWeight[c];
  trial.error = error;
  if (trial.error < best.error) copyCandidate(best, trial);
}

/** Partitioned modes 0, 1, 2, 3 and 7 over the best-ranked `candidates` partitions. */
function tryPartitioned(mode: number, candidates: number): void {
  const info = MODES[mode];
  const ns = info.subsets, partitions = 1 << info.partitionBits, chN = info.alphaBits ? 4 : 3;
  for (let i = 0; i < 64; i++) block[i] = original[i];
  let order: Int32Array;
  if (ns === 2) {
    if (!estimated2) { rankPartitions(2, chN, estimates2, order2); estimated2 = true; }
    order = order2;
  } else {
    if (!estimated3) { rankPartitions(3, chN, estimates3, order3); estimated3 = true; }
    order = order3;
  }
  const levels = 1 << info.indexBits;
  const pbitMode = info.endpointPBits ? 1 : info.sharedPBits ? 2 : 0;
  let tried = 0;
  for (let k = 0; k < 64 && tried < candidates; k++) {
    const partition = order[k];
    if (partition >= partitions) continue; // mode 0 addresses only the first 16 three-subset partitions
    tried++;
    trial.mode = mode; trial.partition = partition; trial.rotation = 0; trial.selection = 0;
    trial.endpoints.fill(0); trial.pbits.fill(0);
    let error = 0;
    for (let s = 0; s < ns && error < best.error; s++) {
      let count = 0;
      for (let i = 0; i < 16; i++) if (SUBSET[ns][partition * 16 + i] === s) list[count++] = i;
      error += fitSubset(count, 0, chN, levels, info.colorBits, info.alphaBits, pbitMode, false, 1);
      for (let e = 0; e < 2; e++) {
        for (let c = 0; c < 4; c++) trial.endpoints[(2 * s + e) * 4 + c] = bestQe[e * 4 + c];
        trial.pbits[2 * s + e] = bestQp[e];
      }
      for (let n = 0; n < count; n++) trial.primary[list[n]] = bestIdx[list[n]];
    }
    trial.error = error;
    if (trial.error < best.error) copyCandidate(best, trial);
  }
}

/** Swaps endpoints so every anchor index has a zero MSB, then writes the 128-bit block. */
const packEndpoints = new Int32Array(24), packPbits = new Int32Array(6);
const packPrimary = new Uint8Array(16), packSecondary = new Uint8Array(16);
let packOut: Uint8Array = new Uint8Array(16), packOffset = 0, packBit = 0;
function write(value: number, count: number): void {
  for (let i = 0; i < count; i++, packBit++) packOut[packOffset + (packBit >> 3)] |= ((value >> i) & 1) << (packBit & 7);
}

function packBlock(cand: Candidate, out: Uint8Array, offset: number): void {
  const info = MODES[cand.mode], ns = info.subsets;
  const endpoints = packEndpoints, pbits = packPbits, primary = packPrimary, secondary = packSecondary;
  endpoints.set(cand.endpoints); pbits.set(cand.pbits); primary.set(cand.primary); secondary.set(cand.secondary);
  const primaryLevels = 1 << info.indexBits;
  const table = ns > 1 ? SUBSET[ns] : null;
  const colorInPrimary = !(info.indexBits2 && cand.selection);
  for (let s = 0; s < ns; s++) {
    const anchor = s === 0 ? 0 : ANCHOR[ns][cand.partition * 3 + s];
    if (primary[anchor] >= primaryLevels >> 1) {
      // Primary indices control all channels unless a secondary set exists.
      const c0 = info.indexBits2 ? (colorInPrimary ? 0 : 3) : 0, c1 = info.indexBits2 ? (colorInPrimary ? 3 : 4) : 4;
      for (let c = c0; c < c1; c++) { const t = endpoints[2 * s * 4 + c]; endpoints[2 * s * 4 + c] = endpoints[(2 * s + 1) * 4 + c]; endpoints[(2 * s + 1) * 4 + c] = t; }
      if (!info.indexBits2) { const t = pbits[2 * s]; pbits[2 * s] = pbits[2 * s + 1]; pbits[2 * s + 1] = t; }
      for (let i = 0; i < 16; i++) if ((table ? table[cand.partition * 16 + i] : 0) === s) primary[i] = primaryLevels - 1 - primary[i];
    }
  }
  if (info.indexBits2) {
    const levels2 = 1 << info.indexBits2;
    if (secondary[0] >= levels2 >> 1) {
      const c0 = colorInPrimary ? 3 : 0, c1 = colorInPrimary ? 4 : 3;
      for (let c = c0; c < c1; c++) { const t = endpoints[c]; endpoints[c] = endpoints[4 + c]; endpoints[4 + c] = t; }
      for (let i = 0; i < 16; i++) secondary[i] = levels2 - 1 - secondary[i];
    }
  }
  out.fill(0, offset, offset + 16);
  packOut = out; packOffset = offset; packBit = 0;
  write(1 << cand.mode, cand.mode + 1);
  write(cand.partition, info.partitionBits);
  write(cand.rotation, info.rotationBits);
  write(cand.selection, info.indexSelectionBits);
  for (let c = 0; c < 3; c++) for (let e = 0; e < 2 * ns; e++) write(endpoints[e * 4 + c], info.colorBits);
  if (info.alphaBits) for (let e = 0; e < 2 * ns; e++) write(endpoints[e * 4 + 3], info.alphaBits);
  if (info.endpointPBits) for (let e = 0; e < 2 * ns; e++) write(pbits[e], 1);
  if (info.sharedPBits) for (let s = 0; s < ns; s++) write(pbits[2 * s], 1);
  for (let i = 0; i < 16; i++) {
    const s = table ? table[cand.partition * 16 + i] : 0;
    const anchor = s === 0 ? i === 0 : ANCHOR[ns][cand.partition * 3 + s] === i;
    write(primary[i], info.indexBits - (anchor ? 1 : 0));
  }
  if (info.indexBits2) for (let i = 0; i < 16; i++) write(secondary[i], info.indexBits2 - (i === 0 ? 1 : 0));
}

/**
 * Encodes 16 RGBA8 pixels (row-major) into one BC7 block at out[offset].
 * Returns the weighted squared error of the chosen encoding. Blocks whose
 * alpha is entirely 255 keep decoded alpha exactly 255.
 */
export function encodeBc7Block(pixels: ArrayLike<number>, out: Uint8Array, offset: number, options: Bc7EncodeOptions = {}): number {
  const quality = options.quality ?? 1, earlyOut = options.earlyOutError ?? 0;
  const weights = options.channelWeights ?? [1, 1, 1, 1];
  let opaque = true;
  for (let i = 0; i < 64; i++) {
    original[i] = pixels[i]; block[i] = pixels[i];
    if ((i & 3) === 3 && pixels[i] !== 255) opaque = false;
  }
  for (let c = 0; c < 4; c++) baseWeight[c] = weights[c];
  if (opaque) baseWeight[3] = OPAQUE_ALPHA_WEIGHT;
  // A zero alpha weight means the material never reads alpha: the opaque-only
  // modes 0-3 (decoded alpha 255) are then as valid as the alpha modes.
  const alphaFree = opaque || weights[3] === 0;
  weight.set(baseWeight);
  estimated2 = false; estimated3 = false;
  best.error = Infinity;
  tryMode6();
  const more = (): boolean => best.error > earlyOut;
  if (quality >= 1 && more()) {
    if (alphaFree) {
      // Rotated separate-channel modes first: cheap, and they win on weakly
      // correlated data channels, which lets the early-out skip partitions.
      for (let rotation = 1; rotation <= 3 && more(); rotation++) {
        tryMode45(5, rotation, 0);
        if (quality >= 2) { tryMode45(4, rotation, 0); tryMode45(4, rotation, 1); }
      }
      if (more()) tryPartitioned(1, quality >= 2 ? 12 : 4);
      if (more()) tryPartitioned(3, quality >= 2 ? 6 : 2);
      if (quality >= 2 && more()) tryPartitioned(0, 4);
      if (quality >= 2 && more()) tryPartitioned(2, 6);
    } else {
      for (let rotation = 0; rotation <= 3 && more(); rotation++) {
        tryMode45(5, rotation, 0);
        if (quality >= 2) { tryMode45(4, rotation, 0); tryMode45(4, rotation, 1); }
      }
      if (more()) tryPartitioned(7, quality >= 2 ? 12 : 4);
    }
  }
  packBlock(best, out, offset);
  return best.error;
}

/**
 * Encodes block rows [rowStart, rowEnd) of an RGBA8 image. Partial edge blocks
 * replicate the last valid row/column (never read outside the image).
 */
export function encodeBc7Rows(pixels: Uint8Array, width: number, height: number, rowStart: number, rowEnd: number,
  options: Bc7EncodeOptions = {}): Uint8Array {
  const bw = Math.ceil(width / 4);
  const out = new Uint8Array((rowEnd - rowStart) * bw * 16), tile = new Uint8Array(64);
  for (let by = rowStart; by < rowEnd; by++) {
    for (let bx = 0; bx < bw; bx++) {
      for (let y = 0; y < 4; y++) {
        const sy = Math.min(height - 1, by * 4 + y);
        for (let x = 0; x < 4; x++) {
          const sx = Math.min(width - 1, bx * 4 + x), s = (sy * width + sx) * 4, t = (y * 4 + x) * 4;
          tile[t] = pixels[s]; tile[t + 1] = pixels[s + 1]; tile[t + 2] = pixels[s + 2]; tile[t + 3] = pixels[s + 3];
        }
      }
      encodeBc7Block(tile, out, ((by - rowStart) * bw + bx) * 16, options);
    }
  }
  return out;
}

/** Single-threaded whole-level encoder (tests and small images). */
export function encodeBc7Image(pixels: Uint8Array, width: number, height: number, options: Bc7EncodeOptions = {}): Uint8Array {
  if (pixels.byteLength !== width * height * 4) throw new RangeError('BC7: expected RGBA8 pixels');
  return encodeBc7Rows(pixels, width, height, 0, Math.ceil(height / 4), options);
}

/** Mode histogram of an encoded level, for build reports. */
export function bc7ModeHistogram(blocks: Uint8Array): number[] {
  const histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let offset = 0; offset < blocks.byteLength; offset += 16) {
    const first = blocks[offset];
    let mode = 0;
    while (mode < 8 && ((first >> mode) & 1) === 0) mode++;
    histogram[mode]++;
  }
  return histogram;
}
