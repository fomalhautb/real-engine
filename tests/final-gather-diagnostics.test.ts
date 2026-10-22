import { describe, expect, it } from 'vitest';
import { decodeFinalGatherDiagnostics, decodeFinalGatherDiagnosticSurfaces, FINAL_GATHER_DIAGNOSTIC_MAX_POINTS,
  validateFinalGatherDiagnosticPoints } from '../src/final-gather-diagnostics';

function record(): Uint32Array {
  const bytes = new ArrayBuffer(32), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
  words[0] = 2 | (3 << 8) | (7 << 16);
  floats.set([1.25, 0.5, 0.75], 1);
  words.set([0x80030000, 0x00000000, 0x00000000, 0x00000000], 4);
  return words;
}

describe('Final Gather diagnostic CPU contract', () => {
  it('decodes exact integer surface identities and signed normals from a subview', () => {
    const backing = new ArrayBuffer(96), words = new Uint32Array(backing, 32, 16), floats = new Float32Array(backing, 32, 16);
    floats.set([1.5, -2, 0.125]);
    words.set([0xfffffff0, 0x7fff0000, 0x7fff7fff, 0xfffffffe, 0xffffffff], 3);
    const decoded = decodeFinalGatherDiagnosticSurfaces(words, [{ x: 0, y: 0 }, { x: 1, y: 0 }]);
    expect(decoded[0]).toEqual({ position: [1.5, -2, 0.125], sourceId: 0xfffffff0, normal: [0, 1, 0],
      geometricNormal: [0, 0, -1], entityId: 0xfffffffe, representationToken: 0xffffffff });
    expect(decoded[1]).toBeNull();
    floats[0] = Infinity;
    expect(() => decodeFinalGatherDiagnosticSurfaces(words, [{ x: 0, y: 0 }, { x: 1, y: 0 }])).toThrow(/POSITION/);
    words.fill(0); words[8] = 1;
    expect(() => decodeFinalGatherDiagnosticSurfaces(words, [{ x: 0, y: 0 }, { x: 1, y: 0 }])).toThrow(/BACKGROUND/);
    expect(() => decodeFinalGatherDiagnosticSurfaces(words, [{ x: 0, y: 0 }])).toThrow(/LENGTH/);
  });
  it('validates a bounded pixel request and copies both list and points', () => {
    const points = [{ x: 0, y: 1 }, { x: 32767, y: 32767 }];
    const copy = validateFinalGatherDiagnosticPoints(points, 32768, 32768);
    expect(copy).toEqual(points);
    expect(copy).not.toBe(points);
    expect(copy[0]).not.toBe(points[0]);
    points[0].x = 7;
    expect(copy[0].x).toBe(0);
    expect(validateFinalGatherDiagnosticPoints(Array.from({ length: FINAL_GATHER_DIAGNOSTIC_MAX_POINTS },
      (_, x) => ({ x, y: 0 })), 64, 1)).toHaveLength(64);
  });

  it('rejects malformed dimensions, counts, positions and duplicates', () => {
    const good = [{ x: 0, y: 0 }];
    for (const size of [0, 32769, 1.5, NaN, Infinity]) {
      expect(() => validateFinalGatherDiagnosticPoints(good, size, 2)).toThrow(/FRAME_SIZE/);
      expect(() => validateFinalGatherDiagnosticPoints(good, 2, size)).toThrow(/FRAME_SIZE/);
    }
    expect(() => validateFinalGatherDiagnosticPoints([], 1, 1)).toThrow(/POINT_COUNT/);
    expect(() => validateFinalGatherDiagnosticPoints(Array.from({ length: 65 }, () => ({ x: 0, y: 0 })), 1, 1))
      .toThrow(/POINT_COUNT/);
    for (const point of [{ x: -1, y: 0 }, { x: 2, y: 0 }, { x: 0, y: 2 },
      { x: 0.5, y: 0 }, { x: NaN, y: 0 }]) {
      expect(() => validateFinalGatherDiagnosticPoints([point], 2, 2)).toThrow(/POINT/);
    }
    expect(() => validateFinalGatherDiagnosticPoints([good[0], { ...good[0] }], 2, 2)).toThrow(/DUPLICATE/);
  });

  it('decodes packed branch, counts, floats and sixteen rejection masks', () => {
    const source = record();
    const decoded = decodeFinalGatherDiagnostics(source, [{ x: 10, y: 12 }])[0];
    expect(decoded).toMatchObject({ pixel: { x: 10, y: 12 }, branch: 'ring', primaryAccepted: 3,
      ringAccepted: 7, totalWeight: 1.25, selectedLuminance: 0.5, worldLuminance: 0.75 });
    expect(decoded.candidateRejections).toHaveLength(16);
    expect(decoded.candidateRejections[0]).toEqual({ mask: 0, reasons: [] });
    expect(decoded.candidateRejections[2]).toEqual({ mask: 3, reasons: ['invalid-or-stale', 'identity'] });
    expect(decoded.candidateRejections[3]).toEqual({ mask: 128, reasons: ['not-visited'] });
    expect(decoded.pixel).not.toBe(source);
  });

  it('rejects malformed record length, header, floats and exclusive rejection masks', () => {
    const pixel = [{ x: 0, y: 0 }];
    expect(() => decodeFinalGatherDiagnostics(new Uint32Array(7), pixel)).toThrow(/LENGTH/);
    for (const header of [6, 1 << 24, 5 << 8, 13 << 16]) {
      const words = record(); words[0] = header;
      expect(() => decodeFinalGatherDiagnostics(words, pixel)).toThrow(/HEADER/);
    }
    for (const value of [-1, Infinity, NaN]) {
      const words = record(); new Float32Array(words.buffer)[1] = value;
      expect(() => decodeFinalGatherDiagnostics(words, pixel)).toThrow(/FLOAT/);
    }
    for (const mask of [65, 129, 192]) {
      const words = record(); words[4] = mask;
      expect(() => decodeFinalGatherDiagnostics(words, pixel)).toThrow(/REJECTION_MASK/);
    }
  });

  it('accepts background and unlit clear records only with zero values and unvisited candidates', () => {
    for (const [branchId, branch] of [[4, 'background'], [5, 'unlit']] as const) {
      const words = new Uint32Array(8);
      words[0] = branchId;
      words.fill(0x80808080, 4);
      expect(decodeFinalGatherDiagnostics(words, [{ x: 1, y: 2 }])[0]).toMatchObject({ branch,
        primaryAccepted: 0, ringAccepted: 0, totalWeight: 0, selectedLuminance: 0, worldLuminance: 0 });
      const nonzeroCount = words.slice(); nonzeroCount[0] |= 1 << 8;
      expect(() => decodeFinalGatherDiagnostics(nonzeroCount, [{ x: 1, y: 2 }])).toThrow(/NON_SURFACE/);
      const nonzeroValue = words.slice(); new Float32Array(nonzeroValue.buffer)[3] = 0.5;
      expect(() => decodeFinalGatherDiagnostics(nonzeroValue, [{ x: 1, y: 2 }])).toThrow(/NON_SURFACE/);
      const visited = words.slice(); visited[4] = 0x80808000;
      expect(() => decodeFinalGatherDiagnostics(visited, [{ x: 1, y: 2 }])).toThrow(/NON_SURFACE/);
    }
  });
});
