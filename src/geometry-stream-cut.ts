import type { Mat4 } from './math';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory, GeometryStreamGroup } from './geometry-stream-types';

export interface GeometryStreamView {
  eye: readonly [number, number, number];
  projectionScale: number;
  near: number;
  maxErrorPixels: number;
  /** Column-major affine object transform; omitted means identity. */
  model?: Mat4;
}
export interface GeometryStreamPlan {
  selectedClusters: number[];
  selectedPages: number[];
  requestedPages: { pageId: number; priority: number }[];
  refinedGroups: number[];
  fallbackGroups: number[];
}

const finite32 = (value: number) => Number.isFinite(value) && Number.isFinite(Math.fround(value));

function checkView(view: GeometryStreamView): void {
  if (!view || !Array.isArray(view.eye) && !(view.eye instanceof Float32Array)
    || view.eye.length !== 3 || !Array.from(view.eye).every(finite32)
    || !finite32(view.projectionScale) || view.projectionScale <= 0
    || !finite32(view.near) || view.near <= 0
    || !finite32(view.maxErrorPixels) || view.maxErrorPixels < 0) throw new RangeError('GEOMETRY_STREAM_VIEW');
  const model = view.model;
  if (model && (model.length !== 16 || !Array.from(model).every(finite32)
    || model[3] !== 0 || model[7] !== 0 || model[11] !== 0 || model[15] !== 1)) {
    throw new RangeError('GEOMETRY_STREAM_MODEL');
  }
}

/** CPU reference for a complete resident cut; page availability must mean GPU-uploaded residency. */
export class GeometryStreamCut {
  readonly directory: GeometryStreamDirectory;
  private readonly descendingGroups: number[];

  constructor(directory: GeometryStreamDirectory) {
    this.directory = validateGeometryStreamDirectory(directory);
    this.descendingGroups = this.directory.groups.map((_, id) => id)
      .sort((a, b) => this.directory.groups[b].level - this.directory.groups[a].level || a - b);
  }

  plan(residentPages: ReadonlySet<number>, view: GeometryStreamView): GeometryStreamPlan {
    checkView(view);
    if (!residentPages || typeof residentPages.has !== 'function') throw new RangeError('GEOMETRY_STREAM_RESIDENCY');
    for (const rootPage of this.directory.rootPages) {
      if (!residentPages.has(rootPage)) throw new Error(`GEOMETRY_STREAM_ROOT_PAGE_MISSING:${rootPage}`);
    }
    const selected = new Set(this.directory.roots);
    const requests = new Map<number, number>();
    const refinedGroups: number[] = [], fallbackGroups: number[] = [];
    for (const groupId of this.descendingGroups) {
      const group = this.directory.groups[groupId];
      if (group.terminal || !group.parents.every(id => selected.has(id))) continue;
      if (!this.needsRefinement(group, view)) continue;
      const missing = new Set<number>();
      for (const child of group.children) {
        const page = this.directory.clusterPages[child];
        if (!residentPages.has(page)) missing.add(page);
      }
      if (missing.size) {
        fallbackGroups.push(groupId);
        for (const page of missing) requests.set(page, Math.max(requests.get(page) ?? -1, group.level));
        continue;
      }
      for (const parent of group.parents) selected.delete(parent);
      for (const child of group.children) selected.add(child);
      refinedGroups.push(groupId);
    }
    const selectedClusters = [...selected].sort((a, b) => a - b);
    const selectedPages = [...new Set(selectedClusters.map(id => this.directory.clusterPages[id]))].sort((a, b) => a - b);
    const requestedPages = [...requests].map(([pageId, priority]) => ({ pageId, priority }))
      .sort((a, b) => b.priority - a.priority || a.pageId - b.pageId);
    return { selectedClusters, selectedPages, requestedPages, refinedGroups, fallbackGroups };
  }

  private needsRefinement(group: GeometryStreamGroup, view: GeometryStreamView): boolean {
    const { min, max } = group.bounds;
    const model = view.model;
    let center = [0, 0, 0], extent = [0, 0, 0], scale = 1;
    if (model) {
      const localCenter = min.map((v, axis) => (v + max[axis]) * 0.5);
      const localExtent = min.map((v, axis) => (max[axis] - v) * 0.5);
      for (let axis = 0; axis < 3; axis++) {
        center[axis] = model[12 + axis] + model[axis] * localCenter[0]
          + model[4 + axis] * localCenter[1] + model[8 + axis] * localCenter[2];
        extent[axis] = Math.abs(model[axis]) * localExtent[0]
          + Math.abs(model[4 + axis]) * localExtent[1] + Math.abs(model[8 + axis]) * localExtent[2];
      }
      const columnSum = Math.max(...[0, 1, 2].map(col =>
        Math.abs(model[col * 4]) + Math.abs(model[col * 4 + 1]) + Math.abs(model[col * 4 + 2])));
      const rowSum = Math.max(...[0, 1, 2].map(row =>
        Math.abs(model[row]) + Math.abs(model[4 + row]) + Math.abs(model[8 + row])));
      scale = Math.sqrt(columnSum * rowSum);
    } else {
      center = min.map((v, axis) => (v + max[axis]) * 0.5);
      extent = min.map((v, axis) => (max[axis] - v) * 0.5);
    }
    // The shader expands world bounds even for an implicit identity transform.
    for (let axis = 0; axis < 3; axis++) extent[axis] += (Math.abs(center[axis]) + extent[axis] + 1) * 0.000002;
    if (!center.every(finite32) || !extent.every(finite32) || !finite32(scale)) {
      throw new RangeError('GEOMETRY_STREAM_WORLD_BOUNDS');
    }
    const distance = Math.hypot(...center.map((value, axis) => Math.max(Math.abs(view.eye[axis] - value) - extent[axis], 0)));
    const pixels = group.error * scale * view.projectionScale / Math.max(view.near, distance);
    return pixels > view.maxErrorPixels;
  }
}
