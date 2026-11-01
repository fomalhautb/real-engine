import { describe, expect, it } from 'vitest';
import { buildGeometryHierarchy, selectGeometryCut } from '../src/geometry-hierarchy';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import { GeometryStreamCut } from '../src/geometry-stream-cut';
import { geometryCutCoverage, geometryCutWatertightness } from '../src/geometry-stream-coverage';
import { g01Mesh, G01_SEED } from '../src/geometry-g01-scene';
import {
  LOD_LABEL_BACKGROUND, LOD_LABEL_OBJECT, LOD_LABEL_UNKNOWN, LodClusterSurface, LodCutCoverageChecker, LodCutEdgeLedger,
  lodCompareCuts, lodCompareSilhouette, lodCoverageLabels, lodCutDifference, lodDistribution, lodGroupErrorPixels, lodLabelDifference,
  lodLabelPopping, lodPixelRay, lodPixelSecant, lodReferenceSilhouette, lodSortedCut, lodSquaredDistanceToLabel, type Vec3,
} from '../src/geometry-lod-error';

function lcg(seed: number) {
  let state = seed >>> 0;
  return () => (state = (Math.imul(state, 1103515245) + 12345) >>> 0) / 4294967296;
}

async function build() {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'lod-error/g01', mesh: g01Mesh(G01_SEED, 64, 16), coverage: 'opaque' },
    { maxClusterTriangles: 32, groupSize: 4, simplifyRatio: 0.5, maxLevels: 12 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 8192 });
  return { hierarchy, directory: await buildGeometryStreamDirectory(hierarchy, encoded) };
}

function bruteSquared(labels: Uint8Array, width: number, height: number, target: number, outside: boolean) {
  const result = new Float64Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let best = Infinity;
    for (let qy = -1; qy <= height; qy++) for (let qx = -1; qx <= width; qx++) {
      const inside = qx >= 0 && qy >= 0 && qx < width && qy < height;
      if (inside ? labels[qy * width + qx] !== target : !outside) continue;
      best = Math.min(best, (qx - x) ** 2 + (qy - y) ** 2);
    }
    result[y * width + x] = best;
  }
  return result;
}

describe('image measurements', () => {
  it('computes exact squared distance transforms, with and without the outside ring', () => {
    const random = lcg(5);
    for (const [width, height] of [[1, 1], [13, 9], [7, 17], [31, 4]]) for (const density of [0, 0.03, 0.4]) {
      const labels = Uint8Array.from({ length: width * height }, () => random() < density ? LOD_LABEL_OBJECT : random() < 0.1 ? LOD_LABEL_UNKNOWN : LOD_LABEL_BACKGROUND);
      for (const outside of [true, false]) {
        const fast = lodSquaredDistanceToLabel(labels, width, height, LOD_LABEL_OBJECT, outside);
        const slow = bruteSquared(labels, width, height, LOD_LABEL_OBJECT, outside);
        for (let p = 0; p < fast.length; p++) {
          if (Number.isFinite(slow[p])) expect(fast[p]).toBe(slow[p]);
          else expect(fast[p]).toBeGreaterThan(1e19);
        }
      }
    }
  });

  it('labels identity colours and measures silhouette displacement against the reference', () => {
    const object = [0.9, 0.42, 0.18], background = [0.0035, 0.0055, 0.0065];
    const rgb = new Float32Array([...object, ...background, 0.5, 0.5, 0.5, 0.9 + 0.01, 0.42, 0.18 - 0.01]);
    expect([...lodCoverageLabels(rgb, object, background)]).toEqual([LOD_LABEL_OBJECT, LOD_LABEL_BACKGROUND, LOD_LABEL_UNKNOWN, LOD_LABEL_OBJECT]);
    // Reference: a disc of radius 6 in a 32x24 image; LOD: the same disc shifted 2 px right.
    const width = 32, height = 24, disc = (cx: number) => Uint8Array.from({ length: width * height }, (_, p) =>
      Math.hypot(p % width - cx, Math.floor(p / width) - 12) <= 6 ? LOD_LABEL_OBJECT : LOD_LABEL_BACKGROUND);
    const reference = lodReferenceSilhouette(disc(14), width, height), lod = disc(16);
    const comparison = lodCompareSilhouette(reference, lod);
    expect(comparison.mismatched).toBe(comparison.lodOnly + comparison.referenceOnly);
    expect(comparison.lodOnly).toBe(comparison.referenceOnly);
    expect(comparison.lodOnly).toBeGreaterThan(0);
    expect(Math.max(...comparison.distances)).toBeLessThanOrEqual(2);
    const toObject = bruteSquared(reference.labels, width, height, LOD_LABEL_OBJECT, true);
    const toBackground = bruteSquared(reference.labels, width, height, LOD_LABEL_BACKGROUND, true);
    comparison.pixels.forEach((p, i) => expect(comparison.distances[i]).toBeCloseTo(Math.sqrt(reference.labels[p] === LOD_LABEL_OBJECT ? toBackground[p] : toObject[p]), 6));
    // The out-of-image ring is a candidate opposite label: a border mismatch is at most one pixel from it.
    const edge = lodReferenceSilhouette(new Uint8Array(width * height).fill(LOD_LABEL_OBJECT), width, height);
    const hole = new Uint8Array(width * height).fill(LOD_LABEL_OBJECT); hole[0] = LOD_LABEL_BACKGROUND; hole[12 * width + 16] = LOD_LABEL_BACKGROUND;
    const holes = lodCompareSilhouette(edge, hole);
    expect([...holes.distances]).toEqual([1, 12]);
  });

  it('counts LOD-only popping, label differences and percentiles', () => {
    const O = LOD_LABEL_OBJECT, B = LOD_LABEL_BACKGROUND, U = LOD_LABEL_UNKNOWN;
    const previousLod = Uint8Array.of(O, O, B, B, O, U), lod = Uint8Array.of(B, O, O, B, B, O);
    const previousReference = Uint8Array.of(O, O, B, O, B, O), reference = Uint8Array.of(O, O, O, B, B, O);
    expect(lodLabelPopping(previousLod, lod, previousReference, reference)).toEqual({ lodChanged: 3, referenceChanged: 2, lodOnly: 2, referenceOnly: 1 });
    expect(lodLabelDifference(previousLod, lod)).toBe(3);
    expect(lodDistribution([])).toEqual({ count: 0, mean: 0, p50: 0, p90: 0, p99: 0, max: 0 });
    const d = lodDistribution(Array.from({ length: 100 }, (_, i) => 100 - i));
    expect(d).toEqual({ count: 100, mean: 50.5, p50: 50, p90: 90, p99: 99, max: 100 });
  });

  it('builds pixel rays and secants consistent with the view axis', () => {
    const eye: Vec3 = [3, 2, -4], target: Vec3 = [0, 0.5, 0], fov = Math.PI / 3.2;
    const width = 480, height = 320;
    for (const [x, y] of [[0, 0], [239.5, 159.5], [479, 319], [100, 300]]) {
      const { direction, forward } = lodPixelRay(eye, target, x, y, width, height, fov);
      const cos = direction[0] * forward[0] + direction[1] * forward[1] + direction[2] * forward[2];
      expect(1 / cos).toBeCloseTo(lodPixelSecant(x, y, width, height, fov), 10);
    }
    const centre = lodPixelRay(eye, target, 239.5, 159.5, width, height, fov);
    centre.direction.forEach((value, axis) => expect(value).toBeCloseTo(centre.forward[axis], 12));
    // Up in the image is +Y in the world for an unrolled lookAt.
    expect(lodPixelRay([0, 0, 5], [0, 0, 0], 240, 0, width, height, fov).direction[1]).toBeGreaterThan(0);
  });
});

describe('cut structure', () => {
  it('matches geometryCutCoverage on valid, partial, duplicated-level and random cuts', async () => {
    const { hierarchy, directory } = await build(), checker = new LodCutCoverageChecker(hierarchy), planner = new GeometryStreamCut(directory);
    expect(hierarchy.groups.length).toBeGreaterThan(8);
    const random = lcg(11), all = hierarchy.clusters.map((_, id) => id);
    const cuts: number[][] = [all.filter(id => hierarchy.clusters[id].level === 0), hierarchy.roots, all];
    for (const tau of [0.25, 1, 4, 64]) for (const eye of [[0, 0.5, 3], [6, 2, -1], [0, 0, 40]] as Vec3[]) {
      const cut = selectGeometryCut(hierarchy, { eye, projectionScale: 300, near: 0.1, maxErrorPixels: tau });
      cuts.push(cut, cut.filter(() => random() < 0.9), [...cut, ...all.filter(id => !cut.includes(id) && random() < 0.05)]);
      cuts.push(planner.plan(new Set(directory.rootPages), { eye, projectionScale: 300, near: 0.1, maxErrorPixels: tau }).selectedClusters);
    }
    for (let k = 0; k < 20; k++) cuts.push(all.filter(() => random() < 0.3));
    let valid = 0;
    for (const cut of cuts) {
      const expected = geometryCutCoverage(directory, cut);
      expect(checker.check(cut)).toEqual(expected);
      if (!expected.uncovered && !expected.overlapping && !expected.partialGroups) valid++;
    }
    expect(valid).toBeGreaterThanOrEqual(14);
    expect(() => checker.check([0, 0])).toThrow('GEOMETRY_CUT_COVERAGE_DUPLICATE');
    expect(() => checker.check([all.length])).toThrow('GEOMETRY_CUT_COVERAGE_CLUSTER');
  });

  it('keeps incremental edge counts equal to geometryCutWatertightness across cut sequences', async () => {
    const { hierarchy } = await build(), ledger = new LodCutEdgeLedger(hierarchy), random = lcg(23);
    const all = hierarchy.clusters.map((_, id) => id);
    const sequence: number[][] = [];
    for (let d = 40; d > 1.5; d *= 0.8) sequence.push(selectGeometryCut(hierarchy, { eye: [0, 0.3 * d, d], projectionScale: 300, near: 0.1, maxErrorPixels: 1 }));
    sequence.push(hierarchy.roots, all.filter(id => hierarchy.clusters[id].level === 0), []);
    for (let k = 0; k < 10; k++) sequence.push(all.filter(() => random() < 0.25));
    sequence.push(hierarchy.roots);
    let watertight = 0;
    for (const cut of sequence) {
      const expected = geometryCutWatertightness(hierarchy, cut);
      expect(ledger.update(cut)).toEqual(expected);
      if (cut.length && expected.boundaryEdges === 0) watertight++;
    }
    expect(watertight).toBeGreaterThan(5);
    expect(ledger.totals).toEqual(geometryCutWatertightness(hierarchy, hierarchy.roots));
    const lod0 = all.filter(id => hierarchy.clusters[id].level === 0);
    ledger.update(lod0);
    for (const id of lod0.slice(0, 5)) expect(ledger.edgeUse(id, 0)).toEqual({ forward: [1, 1, 1], reverse: [1, 1, 1] });
    ledger.update([lod0[0]]);
    expect(ledger.edgeUse(lod0[0], 0).forward).toEqual([1, 1, 1]);
    expect(Math.min(...ledger.edgeUse(lod0[0], 0).reverse)).toBeLessThanOrEqual(1);
    expect(() => ledger.edgeUse(lod0[0], 1e6)).toThrow('LOD_ERROR_LEDGER_TRIANGLE');
    expect(() => ledger.update([0, 0])).toThrow('LOD_ERROR_LEDGER_DUPLICATE');
  });

  it('compares cuts and classifies threshold-borderline differences', async () => {
    const { hierarchy } = await build();
    const view = { eye: [0, 1, 4] as Vec3, projectionScale: 300, near: 0.1, maxErrorPixels: 1 };
    const cut = lodSortedCut(selectGeometryCut(hierarchy, view));
    expect(lodCompareCuts(hierarchy, cut, cut, view)).toEqual({ extra: 0, missing: 0, borderline: 0, unexplained: 0, examples: [] });
    expect(lodCutDifference(cut, cut)).toEqual({ added: 0, removed: 0 });
    // Moving the threshold across one group's projected error flips it: a borderline difference.
    let flips = 0;
    for (let g = 0; g < hierarchy.groups.length && !flips; g++) {
      const entry = hierarchy.groups[g], pixels = entry.terminal ? 0 : lodGroupErrorPixels(entry, view);
      if (!(pixels > 0)) continue;
      const above = { ...view, maxErrorPixels: pixels * (1 + 1e-6) }, below = { ...view, maxErrorPixels: pixels * (1 - 1e-6) };
      const coarse = lodSortedCut(selectGeometryCut(hierarchy, above)), fine = lodSortedCut(selectGeometryCut(hierarchy, below));
      const difference = lodCutDifference(fine, coarse);
      if (!difference.added && !difference.removed) continue;
      const comparison = lodCompareCuts(hierarchy, coarse, fine, below);
      expect(comparison.extra + comparison.missing).toBe(difference.added + difference.removed);
      expect(comparison).toMatchObject({ unexplained: 0, borderline: difference.added + difference.removed });
      flips++;
    }
    expect(flips).toBe(1);
    // An arbitrary removal is not borderline.
    const broken = cut.slice(1);
    expect(lodCompareCuts(hierarchy, broken, cut, view)).toMatchObject({ missing: 1, unexplained: 1 });
    expect(() => lodSortedCut([3, 3])).toThrow('LOD_ERROR_CUT_DUPLICATE');
  });
});

describe('cluster surface queries', () => {
  it('matches brute-force ray casts and closest-point distances', async () => {
    const { hierarchy } = await build(), surface = new LodClusterSurface(hierarchy), random = lcg(31);
    const lod0 = hierarchy.clusters.flatMap((cluster, id) => cluster.level === 0 ? [id] : []);
    const view = new DataView(hierarchy.source.vertices);
    const position = (vertex: number): Vec3 => [0, 1, 2].map(axis => view.getFloat32(vertex * 80 + axis * 4, true)) as unknown as Vec3;
    const triangles = lod0.flatMap(id => {
      const cluster = hierarchy.clusters[id];
      return Array.from({ length: cluster.triangles.length / 3 }, (_, t) => [0, 1, 2].map(k => position(cluster.vertices[cluster.triangles[t * 3 + k]])));
    });
    const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let hits = 0;
    for (let k = 0; k < 40; k++) {
      const theta = random() * Math.PI * 2, origin: Vec3 = [6 * Math.cos(theta), random() * 3 - 1.5, 6 * Math.sin(theta)];
      const aim: Vec3 = [random() * 2 - 1, random() * 0.6 - 0.3, random() * 2 - 1];
      const length = Math.hypot(...sub(aim, origin)), direction = sub(aim, origin).map(value => value / length) as unknown as Vec3;
      let best = Infinity, bestFront = Infinity, facing = false;
      for (const [a, b, c] of triangles) {
        const e1 = sub(b, a), e2 = sub(c, a), p = cross(direction, e2), det = dot(e1, p);
        if (Math.abs(det) < 1e-14) continue;
        const s = sub(origin, a), u = dot(s, p) / det, q = cross(s, e1), v = dot(direction, q) / det, t = dot(e2, q) / det;
        if (!(u >= 0 && v >= 0 && u + v <= 1 && t > 0.1)) continue;
        // Facing the origin: the geometric normal points against the ray.
        const front = dot(cross(e1, e2), direction) < 0;
        if (t < best) { best = t; facing = front; }
        if (front) bestFront = Math.min(bestFront, t);
      }
      const hit = surface.raycast(lod0, origin, direction, 0.1), frontHit = surface.raycast(lod0, origin, direction, 0.1, true);
      if (Number.isFinite(best)) { hits++; expect(hit!.t).toBeCloseTo(best, 9); expect(hit!.front).toBe(facing); } else expect(hit).toBeNull();
      if (Number.isFinite(bestFront)) { expect(frontHit!.t).toBeCloseTo(bestFront, 9); expect(frontHit!.front).toBe(true); } else expect(frontHit).toBeNull();
      // A closed surface seen from outside: the first hit faces the viewer.
      if (hit) expect(hit.front).toBe(true);
      // Closest point from a point near the surface: compare with every triangle via the hit point.
      const point: Vec3 = [origin[0] * 0.3, origin[1] * 0.3, origin[2] * 0.3];
      const near = surface.distance(lod0, point)!;
      const closest = triangles.reduce((minimum, [a, b, c]) => Math.min(minimum, brutePointTriangle(point, a, b, c)), Infinity);
      expect(near.distance).toBeCloseTo(closest, 9);
    }
    expect(hits).toBeGreaterThan(15);
    expect(surface.raycast([], [0, 0, 5], [0, 0, -1])).toBeNull();
    expect(surface.distance([], [0, 0, 0])).toBeNull();
  });
});

/** Independent closest-point distance by projection onto the plane and the three edges. */
function brutePointTriangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3): number {
  const sub = (x: Vec3, y: Vec3): Vec3 => [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
  const dot = (x: Vec3, y: Vec3) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  const cross = (x: Vec3, y: Vec3): Vec3 => [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  const n = cross(sub(b, a), sub(c, a)), nn = dot(n, n);
  const segment = (p: Vec3, q: Vec3) => {
    const d = sub(q, p), t = Math.max(0, Math.min(1, dot(sub(point, p), d) / dot(d, d)));
    return Math.hypot(...sub(point, [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t]));
  };
  let best = Math.min(segment(a, b), segment(b, c), segment(c, a));
  if (nn > 0) {
    const t = dot(sub(point, a), n) / nn, projected: Vec3 = [point[0] - n[0] * t, point[1] - n[1] * t, point[2] - n[2] * t];
    const inside = [[a, b], [b, c], [c, a]].every(([p, q]) => dot(cross(sub(q, p), sub(projected, p)), n) >= 0);
    if (inside) best = Math.min(best, Math.abs(t) * Math.sqrt(nn));
  }
  return best;
}
