/** Column-major 4×4 matrices acting on column vectors. */
export type Mat4 = Float32Array;

export function identity(): Mat4 {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

/** Returns a × b, so b is applied to a point first. */
export function multiply(a: ArrayLike<number>, b: ArrayLike<number>): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let value = 0;
      for (let k = 0; k < 4; k++) value += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = value;
    }
  }
  return out;
}

export function translation(x: number, y: number, z: number): Mat4 {
  const out = identity();
  out[12] = x;
  out[13] = y;
  out[14] = z;
  return out;
}

export function scaling(x: number, y: number, z: number): Mat4 {
  const out = identity();
  out[0] = x;
  out[5] = y;
  out[10] = z;
  return out;
}

export function rotationX(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

export function rotationY(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
}

export function rotationZ(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

/** Right-handed projection with depth in WebGPU's [0, 1] clip interval. */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const range = near - far;
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / range, -1,
    0, 0, (near * far) / range, 0,
  ]);
}

export function lookAt(
  eye: readonly number[], target: readonly number[], up: readonly number[],
): Mat4 {
  const normalize = (v: [number, number, number]): [number, number, number] => {
    const length = Math.hypot(...v);
    if (length === 0) throw new RangeError('lookAt needs distinct eye/target and a usable up vector');
    return [v[0] / length, v[1] / length, v[2] / length];
  };
  const cross = (a: readonly number[], b: readonly number[]): [number, number, number] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1,
  ]);
}

/** Applies the homogeneous transform, including perspective division. */
export function transformPoint(m: ArrayLike<number>, p: readonly number[]): [number, number, number] {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

/** Converts an (x, y, z, w) quaternion to a rotation matrix. */
export function quaternionMatrix(x: number, y: number, z: number, w: number): Mat4 {
  const length = Math.hypot(x, y, z, w);
  if (length === 0) throw new RangeError('quaternion must be nonzero');
  x /= length; y /= length; z /= length; w /= length;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return new Float32Array([
    1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
    2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
    2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
    0, 0, 0, 1,
  ]);
}

/** Inverse of a nonsingular affine transform, including nonuniform scale/shear. */
export function inverseAffine(m: ArrayLike<number>): Mat4 {
  if (m.length !== 16 || !Array.from(m).every(Number.isFinite) || m[3] !== 0 || m[7] !== 0 || m[11] !== 0 || m[15] !== 1) {
    throw new RangeError('inverseAffine requires a finite affine matrix');
  }
  const a = m[0], b = m[4], c = m[8], d = m[1], e = m[5], f = m[9], g = m[2], h = m[6], i = m[10];
  const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(determinant) || determinant === 0) throw new RangeError('inverseAffine requires a nonsingular matrix');
  const out = new Float32Array([
    e * i - f * h, f * g - d * i, d * h - e * g, 0,
    c * h - b * i, a * i - c * g, b * g - a * h, 0,
    b * f - c * e, c * d - a * f, a * e - b * d, 0,
    0, 0, 0, determinant,
  ].map(value => value / determinant));
  for (let row = 0; row < 3; row++) out[12 + row] = -(out[row] * m[12] + out[4 + row] * m[13] + out[8 + row] * m[14]);
  if (!out.every(Number.isFinite)) throw new RangeError('inverseAffine exceeds finite f32 range');
  return out;
}

export function normalMatrix(model: ArrayLike<number>): Mat4 {
  const inverse = inverseAffine(model), out = identity();
  for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++) out[col * 4 + row] = inverse[row * 4 + col];
  return out;
}
