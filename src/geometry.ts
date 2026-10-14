/** Interleaved vertex: position, normal, joints, weights, morph delta (80 bytes). */
export interface MeshData {
  /** Separate packed UV0 stream; omitted assets receive zero UVs. */
  uvs?: Float32Array;
  /** Separate packed tangent xyzw stream, including handedness in w. */
  tangents?: Float32Array;
  vertices: ArrayBuffer;
  indices: Uint32Array;
  vertexCount: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

type Vec3 = [number, number, number];
type Weights = [number, number, number, number];

class MeshBuilder {
  private readonly uvs: number[] = [];
  private readonly data: number[] = [];
  private readonly triangles: number[] = [];
  private readonly min: Vec3 = [Infinity, Infinity, Infinity];
  private readonly max: Vec3 = [-Infinity, -Infinity, -Infinity];

  vertex(position: Vec3, normal: Vec3, weights: Weights = [1, 0, 0, 0], morph: Vec3 = [0, 0, 0]): number {
    const index = this.data.length / 20;
    this.data.push(
      ...position, 1,
      ...normal, 0,
      0, 1, 2, 3,
      ...weights,
      ...morph, 0,
    );
    this.uvs.push(0, 0);
    for (let axis = 0; axis < 3; axis++) {
      this.min[axis] = Math.min(this.min[axis], position[axis]);
      this.max[axis] = Math.max(this.max[axis], position[axis]);
    }
    return index;
  }

  quad(positions: [Vec3, Vec3, Vec3, Vec3], normal: Vec3, weights: Weights, morph: (p: Vec3) => Vec3): void {
    const start = this.data.length / 20;
    for (let i = 0; i < positions.length; i++) {
      this.vertex(positions[i], normal, weights, morph(positions[i]));
      this.uvs[(start + i) * 2] = i === 1 || i === 2 ? 1 : 0;
      this.uvs[(start + i) * 2 + 1] = i >= 2 ? 1 : 0;
    }
    this.triangles.push(start, start + 1, start + 2, start, start + 2, start + 3);
  }

  /** Six outward-facing quads; dimensions are world-space half extents. */
  box(center: Vec3, half: Vec3, weights: Weights = [1, 0, 0, 0], morph: (p: Vec3) => Vec3 = () => [0, 0, 0]): void {
    const faces: [Vec3, Vec3, Vec3, Vec3][] = [
      [[0, 0, 1], [1, 0, 0], [0, 1, 0], [half[0], half[1], half[2]]],
      [[0, 0, -1], [-1, 0, 0], [0, 1, 0], [half[0], half[1], half[2]]],
      [[1, 0, 0], [0, 0, -1], [0, 1, 0], [half[2], half[1], half[0]]],
      [[-1, 0, 0], [0, 0, 1], [0, 1, 0], [half[2], half[1], half[0]]],
      [[0, 1, 0], [1, 0, 0], [0, 0, -1], [half[0], half[2], half[1]]],
      [[0, -1, 0], [1, 0, 0], [0, 0, 1], [half[0], half[2], half[1]]],
    ];
    for (const [normal, u, v, extents] of faces) {
      const faceCenter: Vec3 = [
        center[0] + normal[0] * half[0],
        center[1] + normal[1] * half[1],
        center[2] + normal[2] * half[2],
      ];
      const du = extents[0], dv = extents[1];
      const point = (a: number, b: number): Vec3 => [
        faceCenter[0] + a * du * u[0] + b * dv * v[0],
        faceCenter[1] + a * du * u[1] + b * dv * v[1],
        faceCenter[2] + a * du * u[2] + b * dv * v[2],
      ];
      this.quad([point(-1, -1), point(1, -1), point(1, 1), point(-1, 1)], normal, weights, morph);
    }
  }

  build(): MeshData {
    const vertices = new ArrayBuffer(this.data.length * 4);
    const view = new DataView(vertices);
    for (let vertex = 0; vertex < this.data.length / 20; vertex++) {
      for (let field = 0; field < 20; field++) {
        const offset = (vertex * 20 + field) * 4;
        if (field >= 8 && field < 12) view.setUint32(offset, this.data[vertex * 20 + field], true);
        else view.setFloat32(offset, this.data[vertex * 20 + field], true);
      }
    }
    return {
      vertices,
      uvs: new Float32Array(this.uvs),
      indices: new Uint32Array(this.triangles),
      vertexCount: this.data.length / 20,
      bounds: { min: this.min, max: this.max },
    };
  }
}

/** A neutral-pose character with shoulders at ±0.55, 1.95 and head pivot at y=2.3. */
export function createCharacter(): MeshData {
  const mesh = new MeshBuilder();
  const root: Weights = [1, 0, 0, 0];
  const head: Weights = [0, 1, 0, 0];
  const leftArm: Weights = [0, 0, 1, 0];
  const rightArm: Weights = [0, 0, 0, 1];
  const headMorph = (p: Vec3): Vec3 => [0.16 * p[0], 0.12 * (p[1] - 2.3), 0.13 * p[2]];
  mesh.box([-0.19, 0.5, 0], [0.16, 0.5, 0.17], root);
  mesh.box([0.19, 0.5, 0], [0.16, 0.5, 0.17], root);
  mesh.box([0, 1.58, 0], [0.48, 0.58, 0.25], root);
  mesh.box([0, 2.21, 0], [0.14, 0.08, 0.14], root);
  mesh.box([0, 2.63, 0], [0.36, 0.37, 0.33], head, headMorph);
  mesh.box([-0.16, 2.7, 0.345], [0.045, 0.045, 0.025], head, headMorph);
  mesh.box([0.16, 2.7, 0.345], [0.045, 0.045, 0.025], head, headMorph);
  mesh.box([0, 2.57, 0.36], [0.065, 0.07, 0.08], head, headMorph);
  mesh.box([-0.68, 1.57, 0], [0.15, 0.39, 0.18], leftArm);
  mesh.box([0.68, 1.57, 0], [0.15, 0.39, 0.18], rightArm);
  return mesh.build();
}

/** Root trunk, flexible upper trunk, and two branches under a blocky canopy. */
export function createTree(): MeshData {
  const mesh = new MeshBuilder();
  mesh.box([0, 0.75, 0], [0.19, 0.75, 0.19]);
  mesh.box([0, 1.85, 0], [0.15, 0.75, 0.15], [0.35, 0.65, 0, 0]);
  mesh.box([0, 2.75, 0], [0.12, 0.35, 0.12], [0, 1, 0, 0]);
  mesh.box([-0.43, 2.06, 0], [0.43, 0.1, 0.1], [0, 0, 1, 0]);
  mesh.box([0.43, 2.48, 0], [0.43, 0.1, 0.1], [0, 0, 0, 1]);
  mesh.box([-0.65, 2.45, 0], [0.42, 0.55, 0.38], [0, 0, 1, 0]);
  mesh.box([0.65, 2.8, 0], [0.42, 0.58, 0.38], [0, 0, 0, 1]);
  mesh.box([0, 3.43, 0], [0.56, 0.57, 0.49], [0, 1, 0, 0]);
  return mesh.build();
}

export function createCube(): MeshData {
  const mesh = new MeshBuilder();
  mesh.box([0, 0, 0], [0.5, 0.5, 0.5]);
  return mesh.build();
}

export function createPlane(): MeshData {
  const mesh = new MeshBuilder();
  mesh.quad(
    [[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5]],
    [0, 1, 0], [1, 0, 0, 0], () => [0, 0, 0],
  );
  return mesh.build();
}
