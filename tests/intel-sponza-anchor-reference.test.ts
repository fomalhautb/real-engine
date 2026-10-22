import { describe, expect, it } from 'vitest';
import type { Renderer } from '../src/renderer';
import type { WorldHit, WorldRay } from '../src/world-query';
import { intelSponzaLocalReference, type IntelSponzaLocalReceiver } from './browser/fixtures/intel-sponza-local-reference';
import { intelSponzaAnchorReference, intelSponzaVisibilityLinks,
  type IntelSponzaAnchorReferenceInput, type IntelSponzaVisibilityLink } from './browser/fixtures/intel-sponza-anchor-reference';

const receiver: IntelSponzaLocalReceiver = { pixel: { x: 1, y: 1 }, position: [0, 0, 0],
  normal: [0, 1, 0], geometricNormal: [0, 1, 0], batch: 'receiver', entityId: 1 };
const input: IntelSponzaAnchorReferenceInput = { assets: {}, cases: [{ label: 'receiver', receiver }], batches: 2,
  lighting: { sky: [0, 0, 0], sunColor: [0, 0, 0], sunDirection: [0, 1, 0], maxDistance: 80 },
  expectedTextures: [], expectedImageBytes: 0, expectedQueryGeometry: [] };

function unlit(value: number): WorldHit {
  return { batch: 'light', entityId: 2, entityGeneration: 0, geometry: { kind: 'source' }, triangleIndex: 0,
    distance: 1, position: [0, 1, 0], normal: [0, -1, 0], geometricNormal: [0, -1, 0], previousPosition: [0, 1, 0],
    uv: [0, 0], baseColor: [value, value, value], emissive: [0, 0, 0], opacity: 1, metallic: 0, roughness: 1,
    shading: 'unlit', historyVersion: 0, topologyVersion: 0 };
}

describe('bounded Sponza anchor reference diagnostics (CPU mocked queries only)', () => {
  it('keeps the default 4096 directions unchanged when the default seed is explicit', async () => {
    const directions: number[][][] = [];
    const renderer = { traceRays: async (rays: readonly WorldRay[]) => {
      directions.push(rays.slice(0, 8).map(ray => [...ray.direction]));
      expect(rays).toHaveLength(4096);
      return { frame: 7, materialRevision: 9, hits: rays.map(() => null) };
    } } as unknown as Renderer;
    const first = await intelSponzaLocalReference(renderer, [receiver], [0.2, 0.25, 0.35], [0, 0, 0], [0, 1, 0], 80);
    const second = await intelSponzaLocalReference(renderer, [receiver], [0.2, 0.25, 0.35], [0, 0, 0], [0, 1, 0], 80,
      { seed: 0x6d2b79f5 });
    expect(directions[0]).toEqual(directions[1]);
    expect(first).toEqual(second);
    expect(first.rows[0].contributions.total.luminance.standardError).toBe(0);
  });

  it('merges sample moments including differences between 4096-sample batch means', async () => {
    const chunks: number[] = [], checkpoints: unknown[] = [];
    const renderer = { traceRays: async (rays: readonly WorldRay[]) => {
      chunks.push(rays.length);
      return { frame: 7, materialRevision: 9, hits: rays.map(() => unlit(chunks.length === 1 ? 1 : 3)) };
    } } as unknown as Renderer;
    const result = await intelSponzaAnchorReference(renderer, input, (_, data) => checkpoints.push(data));
    expect(chunks).toEqual([4096, 4096]);
    expect(checkpoints).toHaveLength(2);
    expect(result.cumulative.map(batch => batch.samplesPerCase)).toEqual([4096, 8192]);
    expect(new Set(result.batches.map(batch => batch.seed)).size).toBe(2);
    expect(result.rows[0].contributions.total.mean).toEqual([2 * Math.PI, 2 * Math.PI, 2 * Math.PI]);
    expect(result.rows[0].contributions.total.luminance.mean).toBeCloseTo(2 * Math.PI, 12);
    expect(result.rows[0].contributions.total.luminance.standardError).toBeCloseTo(Math.PI / Math.sqrt(8191), 12);
    expect(result.rows[0].samples).toBe(8192);
    expect(result.rows[0].primary.traced).toBe(8192);
  });

  it('rejects scene snapshot changes between batches', async () => {
    let call = 0;
    const renderer = { traceRays: async (rays: readonly WorldRay[]) => ({
      frame: ++call, materialRevision: 9, hits: rays.map(() => null),
    }) } as unknown as Renderer;
    await expect(intelSponzaAnchorReference(renderer, input)).rejects.toThrow('SPONZA_ANCHOR_REFERENCE_SNAPSHOT_CHANGED');
  });

  it('rejects oversized batches and invalid seeds before issuing a query', async () => {
    const renderer = { traceRays: () => { throw Error('query must not run'); } } as unknown as Renderer;
    await expect(intelSponzaAnchorReference(renderer, { ...input, batches: 9 })).rejects.toThrow('SPONZA_ANCHOR_REFERENCE_BUDGET');
    await expect(intelSponzaLocalReference(renderer, [receiver], [0, 0, 0], [0, 0, 0], [0, 1, 0], 80,
      { seed: -1 })).rejects.toThrow('SPONZA_LOCAL_REFERENCE_SEED');
  });

  it('traces supplied biased origins with normalized, endpoint-trimmed links and no ignored entity', async () => {
    const links: IntelSponzaVisibilityLink[] = [
      { label: 'bias-3cm', origin: [0.03, 0, 0], destination: [3.03, 4, 0] },
      { label: 'bias-0.1mm', origin: [0.0001, 0, 0], destination: [0.0001, 0, 2] },
    ];
    const hit = unlit(1);
    const calls: WorldRay[][] = [];
    const renderer = { traceRays: async (rays: WorldRay[], options: { textureLod: number }) => {
      calls.push(rays);
      expect(options).toEqual({ textureLod: 0 });
      return { frame: 7, materialRevision: 9, hits: [hit, null] };
    } } as unknown as Renderer;
    const result = await intelSponzaVisibilityLinks(renderer, links, { frame: 7, materialRevision: 9 });
    expect(calls).toHaveLength(1);
    expect(calls[0][0]).toEqual({ origin: [0.03, 0, 0], direction: [0.6, 0.8, 0], tMin: 0.0001, tMax: 4.9999 });
    expect(calls[0][1]).toEqual({ origin: [0.0001, 0, 0], direction: [0, 0, 1], tMin: 0.0001, tMax: 1.9999 });
    expect(result.rows.map(row => row.blocked)).toEqual([true, false]);
    expect(result.rows[0].hit).toBe(hit);
    expect(result.rows[1].hit).toBeNull();
    expect(result.budget).toEqual({ links: 2, maxLinks: 256, maxDistance: 1000, queryCalls: 1, maxRays: 2 });
  });

  it('includes visibility work after the reference and enforces its exact snapshot', async () => {
    const calls: number[] = [];
    const link: IntelSponzaVisibilityLink = { label: 'probe', origin: [0, 0, 0], destination: [0, 2, 0] };
    const renderer = { traceRays: async (rays: readonly WorldRay[]) => {
      calls.push(rays.length);
      return { frame: 7, materialRevision: 9, hits: rays.map(() => rays.length === 1 ? unlit(1) : null) };
    } } as unknown as Renderer;
    const result = await intelSponzaAnchorReference(renderer, { ...input, batches: 1, visibilityLinks: [link] });
    expect(calls).toEqual([4096, 1]);
    expect(result.visibilityLinks!.snapshot).toEqual(result.snapshot);
    expect(result.visibilityLinks!.rows[0].blocked).toBe(true);
    expect(result.budget.maxRays).toBe(8193);
    expect(result.budget.maxTraceCalls).toBe(3);
    expect(result.budget.visibilityLinkRays).toBe(1);
    for (const snapshot of [{ frame: 8, materialRevision: 9 }, { frame: 7, materialRevision: 10 }]) {
      await expect(intelSponzaVisibilityLinks(renderer, [link], snapshot)).rejects.toThrow('SPONZA_VISIBILITY_LINK_SNAPSHOT_CHANGED');
    }
  });

  it('validates every link before any reference query and bounds the list at 256', async () => {
    const renderer = { traceRays: () => { throw Error('query must not run'); } } as unknown as Renderer;
    const link: IntelSponzaVisibilityLink = { label: 'probe', origin: [0, 0, 0], destination: [0, 2, 0] };
    for (const invalid of [
      Array.from({ length: 257 }, (_, index) => ({ ...link, label: String(index) })),
      [link, link], [{ ...link, origin: [NaN, 0, 0] }],
      [{ ...link, destination: [0, 0, 0.0002] }], [{ ...link, destination: [0, 1001, 0] }],
    ]) {
      await expect(intelSponzaAnchorReference(renderer, { ...input, visibilityLinks: invalid as IntelSponzaVisibilityLink[] }))
        .rejects.toThrow(/SPONZA_VISIBILITY_LINK_/);
    }
    let calls = 0;
    const bounded = { traceRays: async (rays: readonly WorldRay[]) => {
      calls++; expect(rays).toHaveLength(256);
      return { frame: 7, materialRevision: 9, hits: rays.map(() => null) };
    } } as unknown as Renderer;
    const result = await intelSponzaVisibilityLinks(bounded,
      Array.from({ length: 256 }, (_, index) => ({ ...link, label: String(index) })), { frame: 7, materialRevision: 9 });
    expect(calls).toBe(1);
    expect(result.rows).toHaveLength(256);
    const empty = await intelSponzaVisibilityLinks(renderer, [], { frame: 7, materialRevision: 9 });
    expect(empty.rows).toEqual([]);
    expect(empty.budget.queryCalls).toBe(0);
  });

  it('does not turn incomplete queries or out-of-segment hits into a visibility verdict', async () => {
    const link: IntelSponzaVisibilityLink = { label: 'probe', origin: [0, 0, 0], destination: [0, 2, 0] };
    const overflow = { traceRays: async () => { throw Error('WORLD_QUERY_WORK_BUDGET'); } } as unknown as Renderer;
    await expect(intelSponzaVisibilityLinks(overflow, [link], { frame: 7, materialRevision: 9 })).rejects.toThrow('WORLD_QUERY_WORK_BUDGET');
    for (const hits of [[], [{ ...unlit(1), distance: 2 }]]) {
      const invalid = { traceRays: async () => ({ frame: 7, materialRevision: 9, hits }) } as unknown as Renderer;
      await expect(intelSponzaVisibilityLinks(invalid, [link], { frame: 7, materialRevision: 9 }))
        .rejects.toThrow(/SPONZA_VISIBILITY_LINK_(HIT_COUNT|HIT_INTERVAL)/);
    }
  });
});
