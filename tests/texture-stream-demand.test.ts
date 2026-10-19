import { describe, expect, it } from 'vitest';
import type { MeshData } from '../src/geometry';
import { identity, perspective, rotationZ, scaling } from '../src/math';
import { meshWorldUnitsPerUv, modelStretch, textureViewDemand } from '../src/texture-stream-demand';

function triangle(uvs: number[] = [0, 0, 1, 0, 0, 1]): MeshData {
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  const positions = [[0, 0, 0], [2, 0, 0], [0, 3, 0]];
  positions.forEach((point, vertex) => point.forEach((value, axis) => view.setFloat32(vertex * 80 + axis * 4, value, true)));
  return { vertices, uvs: new Float32Array(uvs), indices: new Uint32Array([0, 1, 2]), vertexCount: 3,
    bounds: { min: [0, 0, 0], max: [2, 3, 0] } };
}

describe('static mesh UV stretch', () => {
  it('bounds a known affine UV mapping and changes with tiling', () => {
    expect(meshWorldUnitsPerUv(triangle())).toBeCloseTo(Math.sqrt(13));
    expect(meshWorldUnitsPerUv(triangle([0, 0, 2, 0, 0, 3]))).toBeCloseTo(Math.sqrt(2));
  });

  it('uses the maximum over triangles', () => {
    const mesh = triangle();
    const doubled = new ArrayBuffer(6 * 80);
    new Uint8Array(doubled).set(new Uint8Array(mesh.vertices));
    new Uint8Array(doubled).set(new Uint8Array(mesh.vertices), 3 * 80);
    const mixed: MeshData = { ...mesh, vertices: doubled, vertexCount: 6,
      uvs: new Float32Array([0, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 3, 4, 5]) };
    expect(meshWorldUnitsPerUv(mixed)).toBeCloseTo(Math.sqrt(13));
  });

  it('asks for finest quality when UV scale cannot be proved', () => {
    expect(meshWorldUnitsPerUv({ ...triangle(), uvs: undefined })).toBe(Infinity);
    expect(meshWorldUnitsPerUv(triangle([0, 0, 1, 0, 2, 0]))).toBe(Infinity);
    expect(meshWorldUnitsPerUv(triangle([0, 0, NaN, 0, 0, 1]))).toBe(Infinity);
  });

  it('rejects malformed or unbounded geometry before reading it', () => {
    expect(() => meshWorldUnitsPerUv({ ...triangle(), indices: new Uint32Array([0, 1, 3]) })).toThrow();
    expect(() => meshWorldUnitsPerUv({ ...triangle(), vertices: new ArrayBuffer(3) })).toThrow();
    expect(() => meshWorldUnitsPerUv({ ...triangle(), vertexCount: 1_048_577 })).toThrow();
    expect(() => meshWorldUnitsPerUv({ ...triangle(), uvs: new Float32Array(2) })).toThrow();
  });
});

describe('model stretch bound', () => {
  it('is exact for identity, axis scales, and right-angle rotations', () => {
    expect(modelStretch(identity())).toBe(1);
    expect(modelStretch(scaling(2, 3, 4))).toBe(4);
    expect(modelStretch(rotationZ(Math.PI / 2))).toBeCloseTo(1);
  });

  it('conservatively bounds shear and rejects invalid matrices', () => {
    const shear = identity();
    shear[4] = 1;
    expect(modelStretch(shear)).toBeCloseTo(2);
    expect(modelStretch(shear)).toBeGreaterThan(Math.hypot(1, 1));
    expect(() => modelStretch([1, 2])).toThrow();
  });
});

describe('texture view demand', () => {
  const base = {
    bounds: { min: [-0.5, -0.5, -4.5], max: [0.5, 0.5, -3.5] } as MeshData['bounds'],
    viewProjection: perspective(Math.PI / 2, 2, 0.1, 100),
    eye: [0, 0, 0], forward: [0, 0, -1], focalPixels: 100,
    worldUnitsPerUv: 1, width: 400, height: 200, maxMip: 8,
  };

  it('uses the nearest positive AABB depth and viewport size', () => {
    const normal = textureViewDemand(base);
    expect(normal.visible).toBe(true);
    expect(normal.mip).toBe(2); // floor(log2(200 * 3.5 / 100))
    expect(textureViewDemand({ ...base, width: 400, height: 400 }).mip).toBe(3);
    expect(textureViewDemand({ ...base, width: 100, height: 200,
      viewProjection: perspective(Math.PI / 2, 0.5, 0.1, 100) }).mip).toBe(1);
    expect(textureViewDemand({ ...base, worldUnitsPerUv: 2 }).mip).toBe(1);
  });

  it('accounts for perspective stretch away from the view axis', () => {
    const nearAxis = textureViewDemand({ ...base, focalPixels: 90,
      bounds: { min: [-0.01, -0.01, -4.01], max: [0.01, 0.01, -3.99] } });
    const atEdge = textureViewDemand({ ...base, focalPixels: 90,
      bounds: { min: [3.99, -0.01, -4.01], max: [4.01, 0.01, -3.99] } });
    // At x ≈ depth, the perspective Jacobian is about sqrt(2) times larger.
    // The independent ratios are 200*3.99/90 ≈ 8.87 and 8.87/sqrt(2) ≈ 6.27.
    expect(nearAxis.visible).toBe(true);
    expect(atEdge.visible).toBe(true);
    expect(nearAxis.mip).toBe(3);
    expect(atEdge.mip).toBe(2);
  });

  it('culls outside each clip plane and gives no priority to offscreen bounds', () => {
    const right = textureViewDemand({ ...base,
      bounds: { min: [20, -0.5, -4.5], max: [21, 0.5, -3.5] } });
    const behind = textureViewDemand({ ...base,
      bounds: { min: [-0.5, -0.5, 1], max: [0.5, 0.5, 2] } });
    expect(right).toEqual({ visible: false, mip: 8, priority: 0 });
    expect(behind).toEqual({ visible: false, mip: 8, priority: 0 });
  });

  it('requests mip zero at the near plane or with unknown UV scale', () => {
    const near = textureViewDemand({ ...base,
      bounds: { min: [-0.5, -0.5, -1], max: [0.5, 0.5, 0] } });
    const unknown = textureViewDemand({ ...base, worldUnitsPerUv: Infinity });
    expect(near.visible).toBe(true);
    expect(near.mip).toBe(0);
    expect(unknown.mip).toBe(0);
    expect(Number.isFinite(near.priority)).toBe(true);
    expect(Number.isFinite(unknown.priority)).toBe(true);
  });

  it('clamps mip and gives nearer visible objects higher finite priority', () => {
    const near = textureViewDemand(base);
    const far = textureViewDemand({ ...base,
      bounds: { min: [-0.5, -0.5, -40.5], max: [0.5, 0.5, -39.5] } });
    expect(far.mip).toBe(6);
    expect(near.priority).toBeGreaterThan(far.priority);
    expect(textureViewDemand({ ...base, maxMip: 1 }).mip).toBe(1);
    expect(Number.isFinite(far.priority)).toBe(true);
  });
});
