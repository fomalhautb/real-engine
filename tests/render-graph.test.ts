import { describe, expect, it } from 'vitest';
import { compileRenderGraph, type RenderPass, type RenderResource, type RenderResourceUse } from '../src/render-graph';

type Context = { calls: string[] };
const pass = (
  name: string, uses: RenderPass<Context>['uses'], after?: readonly string[],
): RenderPass<Context> => ({ name, uses, after, execute: context => { context.calls.push(name); } });
const buffer = (name: string, source: RenderResource['source'] = 'transient'): RenderResource =>
  ({ name, kind: 'buffer', source });
const texture = (name: string, source: RenderResource['source'] = 'transient'): RenderResource =>
  ({ name, kind: 'texture', source });

describe('render graph', () => {
  it('schedules shared deformation before shadow and surface consumers', () => {
    const graph = compileRenderGraph([buffer('geometry'), texture('shadow')], [
      pass('deform', [{ resource: 'geometry', access: 'write' }]),
      pass('shadow', [
        { resource: 'geometry', access: 'read' }, { resource: 'shadow', access: 'write' },
      ]),
      pass('surface', [
        { resource: 'geometry', access: 'read' }, { resource: 'shadow', access: 'read' },
      ]),
    ]);
    const context: Context = { calls: [] };
    graph.execute(context);
    expect(graph.order).toEqual(['deform', 'shadow', 'surface']);
    expect(context.calls).toEqual(graph.order);
    expect(graph.lifetimes.get('geometry')).toEqual({ first: 0, last: 2 });
    expect(graph.lifetimes.get('shadow')).toEqual({ first: 1, last: 2 });
  });

  it('accepts imported reads and tracks unused resources', () => {
    const graph = compileRenderGraph([buffer('external', 'imported'), buffer('unused')], [
      pass('read', [{ resource: 'external', access: 'read' }]),
    ]);
    expect(graph.order).toEqual(['read']);
    expect(graph.lifetimes.get('external')).toEqual({ first: 0, last: 0 });
    expect(graph.lifetimes.get('unused')).toBeNull();
  });

  it('retains write-after-read order for imported resources', () => {
    const graph = compileRenderGraph([buffer('history', 'imported')], [
      pass('sample', [{ resource: 'history', access: 'read' }]),
      pass('replace', [{ resource: 'history', access: 'write' }]),
    ]);
    expect(graph.order).toEqual(['sample', 'replace']);
    expect(() => compileRenderGraph([buffer('history', 'imported')], [
      pass('sample', [{ resource: 'history', access: 'read' }], ['replace']),
      pass('replace', [{ resource: 'history', access: 'write' }]),
    ])).toThrow(/cycle/);
  });

  it('uses explicit dependencies and stable declaration-order ties', () => {
    const graph = compileRenderGraph([], [
      pass('first', []), pass('consumer', [], ['producer']), pass('producer', []), pass('last', []),
    ]);
    expect(graph.order).toEqual(['first', 'producer', 'consumer', 'last']);
  });

  it('keeps validated uses, dependencies, and callbacks after input mutation', () => {
    const uses: RenderResourceUse[] = [{ resource: 'data', access: 'write' }];
    const after = ['producer'];
    const passes: RenderPass<Context>[] = [
      { name: 'producer', uses, execute: context => { context.calls.push('producer'); } },
      { name: 'consumer', uses: [{ resource: 'data', access: 'read' }], after,
        execute: context => { context.calls.push('consumer'); } },
    ];
    const graph = compileRenderGraph([buffer('data')], passes);
    uses[0].resource = 'missing';
    after[0] = 'consumer';
    passes[0].name = 'changed';
    passes[0].execute = context => { context.calls.push('changed'); };
    passes.reverse();
    passes.pop();

    const context: Context = { calls: [] };
    graph.execute(context);
    expect(graph.order).toEqual(['producer', 'consumer']);
    expect(graph.lifetimes.get('data')).toEqual({ first: 0, last: 1 });
    expect(context.calls).toEqual(['producer', 'consumer']);
  });

  it('rejects a transient read before its first write', () => {
    expect(() => compileRenderGraph([buffer('geometry')], [
      pass('draw', [{ resource: 'geometry', access: 'read' }]),
      pass('deform', [{ resource: 'geometry', access: 'write' }]),
    ])).toThrow(/before its first write/);
    expect(() => compileRenderGraph([buffer('geometry')], [
      pass('mutate', [{ resource: 'geometry', access: 'readWrite' }]),
    ])).toThrow(/before its first write/);
  });

  it('rejects missing names, self dependencies, and cycles', () => {
    expect(() => compileRenderGraph([], [pass('a', [], ['missing'])])).toThrow(/missing pass/);
    expect(() => compileRenderGraph([], [pass('a', [], ['a'])])).toThrow(/depends on itself/);
    expect(() => compileRenderGraph([], [pass('a', [], ['b']), pass('b', [], ['a'])])).toThrow(/cycle/);
    expect(() => compileRenderGraph([], [pass('a', [{ resource: 'missing', access: 'read' }])])).toThrow(/missing resource/);
    expect(() => compileRenderGraph([buffer('x'), buffer('x')], [])).toThrow(/Duplicate/);
    expect(() => compileRenderGraph([], [pass('a', []), pass('a', [])])).toThrow(/Duplicate/);
  });

  it('requires explicit buffer readWrite and rejects texture self hazards', () => {
    expect(() => compileRenderGraph([buffer('data', 'imported')], [
      pass('mutate', [
        { resource: 'data', access: 'read' }, { resource: 'data', access: 'write' },
      ]),
    ])).toThrow(/more than once/);
    expect(() => compileRenderGraph([texture('color', 'imported')], [
      pass('feedback', [{ resource: 'color', access: 'readWrite' }]),
    ])).toThrow(/cannot readWrite texture/);
    const graph = compileRenderGraph([buffer('data', 'imported')], [
      pass('mutate', [{ resource: 'data', access: 'readWrite' }]),
    ]);
    expect(graph.order).toEqual(['mutate']);
  });
});
