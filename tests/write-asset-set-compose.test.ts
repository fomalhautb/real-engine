import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadAssetSetRoots } from '../src/gltf-package-scene';

type Json = Record<string, any>;
const sha256 = (bytes: Uint8Array | string): string => createHash('sha256').update(bytes).digest('hex');
const NOTICE = 'Synthetic fixture notice for write-asset-set compose tests.\n';
const TOTAL_KEYS = ['sourceTriangles', 'triangles', 'removedTriangles', 'pagedPrimitives', 'sourcePrimitives', 'clusters', 'pages',
  'encodedPageBytes', 'decodedPageBytes', 'rootEncodedBytes', 'rootDecodedBytes', 'proxyVertices', 'proxyTriangles', 'proxyBytes'];

function tool(args: string[]): { status: number | null; stdout: string; stderr: string } {
  const child = spawnSync(process.execPath, [resolve('scripts/write-asset-set.mjs'), ...args], {
    cwd: resolve('.'), encoding: 'utf8', timeout: 60_000, windowsHide: true });
  return { status: child.status, stdout: child.stdout, stderr: child.stderr };
}
function succeeds(args: string[]): Json {
  const result = tool(args);
  expect(result.status, result.stderr).toBe(0);
  return JSON.parse(result.stdout);
}
function fails(args: string[], code: string): void {
  const result = tool(args);
  expect(result.status).not.toBe(0);
  expect(result.stderr).toContain(code);
}

const geometryTotals = (k: number): Json => Object.fromEntries(TOTAL_KEYS.map((key, i) => [key, k * (i + 1) * 100 + i]));

interface Fixture { name: string; sourceSha256: string; geometrySha256: string; args: string[]; totals: Json }

/** Writes synthetic geometry/texture/UV(/query/camera) packages of one source under `root`. */
async function packages(root: string, name: string, k: number, options: { query?: boolean; camera?: boolean; uv?: boolean;
  source?: string; dropTotal?: string } = {}): Promise<Fixture> {
  const sourceSha256 = sha256('source-' + (options.source ?? name)), file = `source/${options.source ?? name}.gltf`;
  const totals = geometryTotals(k);
  if (options.dropTotal) delete totals[options.dropTotal];
  const geometry = JSON.stringify({ version: 1, format: 'real-engine-gltf-geometry', source: { file, bytes: 100 + k, sha256: sourceSha256, buffers: [] },
    assetIdentity: sha256('identity-' + name), config: { removeZeroAreaTriangles: true, geometryBuild: { groupSize: 8 * k } }, totals,
    primitives: [{ meshIndex: 0, primitiveIndex: 0, materialIndex: 0, coverage: 'opaque', route: 'paged' },
      { meshIndex: 1, primitiveIndex: 0, materialIndex: 1, coverage: 'blend', route: 'source' }] });
  await mkdir(join(root, `${name}-geometry`));
  await writeFile(join(root, `${name}-geometry`, 'manifest.json'), geometry);
  await mkdir(join(root, `${name}-textures`));
  await writeFile(join(root, `${name}-textures`, 'manifest.json'), JSON.stringify({ version: 1, format: 'real-engine-gltf-texture-set',
    source: { file, sha256: sourceSha256 }, encoder: { channelPolicy: 'material' }, complete: true, images: [],
    totals: { images: 2 * k, bc7Images: k, rgba8Images: k, nativeBytes: 4096 * k } }));
  const args = ['--name', 'test-' + name, '--geometry', join(root, `${name}-geometry`), '--textures', join(root, `${name}-textures`),
    '--license', join(root, 'inputs', 'credits_license.txt'), '--artifacts-root', root];
  if (options.uv !== false) {
    await writeFile(join(root, `${name}-uv.json`), JSON.stringify({ format: 'real-engine-gltf-uv-demand-audit', sourceSha256,
      summary: { primitives: 2, finite: 1, unknown: 1 }, geometryCheck: { manifestSha256: sha256(geometry), identical: true } }));
    args.push('--uv', join(root, `${name}-uv.json`));
  }
  if (options.query) {
    await mkdir(join(root, `${name}-query`));
    await writeFile(join(root, `${name}-query`, 'manifest.json'), JSON.stringify({ format: 'real-engine-gltf-query-proxies',
      sourcePackage: { assetIdentity: sha256('identity-' + name), manifestSha256: sha256(geometry), sourceSha256 },
      config: { maxError: 0 }, totals: { primitives: 1, triangles: 1000 * k } }));
    args.push('--query', join(root, `${name}-query`));
  }
  if (options.camera) {
    await writeFile(join(root, 'inputs', `${name}-camera.json`), JSON.stringify({ format: 'real-engine-camera-path', name: name + ' walk',
      samples: [{ time: 0, position: [0, 1, 0], forward: [1, 0, 0], up: [0, 1, 0] }, { time: 2 * k, position: [k, 1, 0], forward: [0, 0, 2], up: [0, 3, 0] }] }));
    args.push('--camera-path', join(root, 'inputs', `${name}-camera.json`));
  }
  return { name, sourceSha256, geometrySha256: sha256(geometry), args, totals };
}
async function assetSet(root: string, fixture: Fixture, directory = `${fixture.name}-set`): Promise<string> {
  succeeds([...fixture.args, '--output', join(root, directory)]);
  return join(root, directory, 'asset-set.json');
}

// Each test runs the tool as a child process many times, so it gets a 60 s
// limit: the default 5 s is exceeded on a loaded machine.
describe('write-asset-set compose mode', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('composes re-verified single-source sets of distinct sources deterministically', async () => {
    const root = await mkdtemp(join(resolve('.'), '.write-asset-set-compose-test-'));
    try {
      await mkdir(join(root, 'inputs'));
      await writeFile(join(root, 'inputs', 'credits_license.txt'), NOTICE);
      const main = await packages(root, 'main', 1, { query: true, camera: true }), ivy = await packages(root, 'ivy', 2);
      const mainSet = await assetSet(root, main), ivySet = await assetSet(root, ivy);

      // Single-source mode output, field by field (its bytes are what compose mode hashes).
      const single = JSON.parse(await readFile(mainSet, 'utf8'));
      const camera = await readFile(join(root, 'inputs', 'main-camera.json'));
      expect(single).toEqual({ version: 1, format: 'real-engine-asset-set', name: 'test-main',
        source: { file: 'source/main.gltf', bytes: 101, sha256: main.sourceSha256 },
        roots: { geometryRoot: '/artifacts/main-geometry/', textureGpuRoot: '/artifacts/main-textures/', textureReferenceRoot: '/artifacts/main-textures/',
          fineQueryRoot: '/artifacts/main-query/', uvAuditUrl: '/artifacts/main-uv.json', cameraPathUrl: '/artifacts/main-set/camera-path.json' },
        packages: {
          geometry: { root: '/artifacts/main-geometry/', manifestSha256: main.geometrySha256, assetIdentity: sha256('identity-main') },
          textures: { root: '/artifacts/main-textures/', manifestSha256: sha256(await readFile(join(root, 'main-textures', 'manifest.json'))),
            images: 2, bc7Images: 1, rgba8Images: 1, nativeBytes: 4096 },
          fineQuery: { root: '/artifacts/main-query/', manifestSha256: sha256(await readFile(join(root, 'main-query', 'manifest.json'))),
            maxError: 0, totals: { primitives: 1, triangles: 1000 } },
          uvAudit: { url: '/artifacts/main-uv.json', sha256: sha256(await readFile(join(root, 'main-uv.json'))),
            summary: { primitives: 2, finite: 1, unknown: 1 }, geometryChecked: true },
          cameraPath: { url: '/artifacts/main-set/camera-path.json', sha256: sha256(camera), samples: 2, durationSeconds: 2 } },
        license: { file: 'credits_license.txt', bytes: NOTICE.length, sha256: sha256(NOTICE) },
        omitted: [{ meshIndex: 1, primitiveIndex: 0, materialIndex: 1, coverage: 'blend', route: 'source' }] });
      // Single-source readers keep accepting single sets (the scene roots; the fine query root is read elsewhere).
      vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(single))));
      const { fineQueryRoot: _fineQueryRoot, ...sceneRoots } = single.roots;
      await expect(loadAssetSetRoots('/artifacts/main-set/asset-set.json')).resolves.toEqual(sceneRoots);

      const compose = ['--compose', 'main=' + mainSet, '--compose', 'ivy=' + ivySet, '--name', 'test-composed',
        '--license', join(root, 'inputs', 'credits_license.txt'), '--artifacts-root', root];
      const printed = succeeds([...compose, '--output', join(root, 'composed')]);
      expect(printed).toMatchObject({ components: ['main', 'ivy'], cameraPath: 'main' });
      expect((await readdir(join(root, 'composed'))).sort()).toEqual(['asset-set.json', 'credits_license.txt', 'manifest.json']);
      const text = await readFile(join(root, 'composed', 'asset-set.json'), 'utf8');
      expect(await readFile(join(root, 'composed', 'manifest.json'), 'utf8')).toBe(text);
      expect(await readFile(join(root, 'composed', 'credits_license.txt'), 'utf8')).toBe(NOTICE);
      const composition = JSON.parse(text);
      expect(composition).toMatchObject({ version: 1, format: 'real-engine-asset-composition', name: 'test-composed',
        frame: { componentTransform: 'identity' }, license: { file: 'credits_license.txt', bytes: NOTICE.length, sha256: sha256(NOTICE) } });
      expect(composition.components.map((c: Json) => c.name)).toEqual(['main', 'ivy']);
      for (const [component, fixture, file] of [[composition.components[0], main, mainSet], [composition.components[1], ivy, ivySet]] as const) {
        const bytes = await readFile(file), set = JSON.parse(bytes.toString('utf8'));
        expect(component.setName).toBe(set.name);
        expect(component.assetSet).toEqual({ url: `/artifacts/${fixture.name}-set/asset-set.json`, bytes: bytes.length, sha256: sha256(bytes) });
        expect(component.source).toEqual(set.source);
        expect(component.license).toEqual(set.license);
        expect(component.omitted).toEqual(set.omitted);
        expect(component.packages.geometry).toEqual({ root: set.packages.geometry.root, manifestSha256: fixture.geometrySha256,
          assetIdentity: set.packages.geometry.assetIdentity, groupSize: fixture === main ? 8 : 16, removeZeroAreaTriangles: true, totals: fixture.totals });
        expect(component.packages.textures).toEqual({ ...set.packages.textures, channelPolicy: 'material' });
        expect(component.packages.uvAudit).toEqual(set.packages.uvAudit);
        expect(component.packages.fineQuery).toEqual(set.packages.fineQuery);
        expect(component.packages.cameraPath).toEqual(set.packages.cameraPath);
      }
      expect(composition.cameraPath).toEqual({ component: 'main', ...single.packages.cameraPath });
      expect(composition.totals).toMatchObject({ components: 2, omittedPrimitives: 2,
        geometry: Object.fromEntries(TOTAL_KEYS.map(key => [key, main.totals[key] + ivy.totals[key]])),
        textures: { images: 6, bc7Images: 3, rgba8Images: 3, nativeBytes: 3 * 4096 } });

      // Deterministic: the same inputs give the same bytes; published directories are never rewritten.
      succeeds([...compose, '--output', join(root, 'composed-again')]);
      expect(await readFile(join(root, 'composed-again', 'asset-set.json'), 'utf8')).toBe(text);
      fails([...compose, '--output', join(root, 'composed')], 'ASSET_SET_OUTPUT_EXISTS');

      // Single-source readers reject a composition instead of reading one component of it.
      vi.stubGlobal('fetch', vi.fn(async () => new Response(text)));
      await expect(loadAssetSetRoots('/artifacts/composed/asset-set.json')).rejects.toThrow('GLTF_PACKAGE_ASSET_SET_FORMAT');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 60_000);

  it('rejects mixed arguments, duplicates, stale or partial packages, missing audits and foreign licenses', async () => {
    const root = await mkdtemp(join(resolve('.'), '.write-asset-set-compose-test-'));
    try {
      await mkdir(join(root, 'inputs'));
      await writeFile(join(root, 'inputs', 'credits_license.txt'), NOTICE);
      await writeFile(join(root, 'inputs', 'other_license.txt'), NOTICE + 'changed\n');
      const main = await packages(root, 'main', 1, { camera: true }), ivy = await packages(root, 'ivy', 2);
      const mainSet = await assetSet(root, main), ivySet = await assetSet(root, ivy);
      const tail = ['--name', 'test-composed', '--license', join(root, 'inputs', 'credits_license.txt'), '--artifacts-root', root, '--output', join(root, 'out')];
      const both = ['--compose', 'main=' + mainSet, '--compose', 'ivy=' + ivySet];

      fails([...both, ...tail, '--geometry', join(root, 'main-geometry')], 'ASSET_SET_COMPOSE_ARGUMENTS: --geometry');
      fails([...main.args, '--camera-component', 'main', '--output', join(root, 'out')], 'ASSET_SET_ARGUMENTS: --camera-component');
      fails([...both, '--name', 'test-composed', '--artifacts-root', root, '--output', join(root, 'out')], 'ASSET_SET_REQUIRED: --license');
      fails(['--compose', 'Main=' + mainSet, ...tail], 'ASSET_SET_COMPOSE_COMPONENT');
      fails(['--compose', 'main=' + join(root, 'main-set', 'manifest.json'), ...tail], 'ASSET_SET_COMPOSE_COMPONENT: main must name an asset-set.json');
      fails(['--compose', 'main=' + join(root, 'absent', 'asset-set.json'), ...tail], 'ASSET_SET_COMPOSE_MISSING: main asset set');
      fails(['--compose', 'main=' + mainSet, '--compose', 'main=' + ivySet, ...tail], 'ASSET_SET_COMPOSE_DUPLICATE_NAME: main');
      fails(['--compose', 'main=' + mainSet, '--compose', 'again=' + mainSet, ...tail], 'ASSET_SET_COMPOSE_DUPLICATE_SOURCE: again');
      // Another set of the same source (different packages, same source document) is a duplicate too.
      const twin = await packages(root, 'twin', 3, { source: 'main' });
      fails(['--compose', 'main=' + mainSet, '--compose', 'twin=' + await assetSet(root, twin), ...tail], 'ASSET_SET_COMPOSE_DUPLICATE_SOURCE: twin');
      fails([...both, ...tail.slice(0, 2), '--license', join(root, 'inputs', 'other_license.txt'), ...tail.slice(4)], 'ASSET_SET_COMPOSE_LICENSE_MISMATCH: main');

      // Camera choice: one path is taken, two need --camera-component, a named component must have one.
      const second = await packages(root, 'second', 4, { camera: true });
      const secondSet = await assetSet(root, second);
      fails([...both, '--compose', 'second=' + secondSet, ...tail], 'ASSET_SET_COMPOSE_CAMERA_AMBIGUOUS: main,second');
      fails([...both, '--camera-component', 'ivy', ...tail], 'ASSET_SET_COMPOSE_CAMERA: ivy has no camera path');
      const chosen = succeeds([...both, '--compose', 'second=' + secondSet, '--camera-component', 'second', ...tail.slice(0, -1), join(root, 'chosen')]);
      expect(chosen.cameraPath).toBe('second');

      // The scene loader needs a UV audit; totals must be complete counters, never silently zero.
      const bare = await packages(root, 'bare', 5, { uv: false });
      fails(['--compose', 'bare=' + await assetSet(root, bare), ...tail], 'ASSET_SET_COMPOSE_UV_REQUIRED: bare');
      const partial = await packages(root, 'partial', 6, { dropTotal: 'proxyVertices' });
      fails(['--compose', 'partial=' + await assetSet(root, partial), ...tail], 'ASSET_SET_COMPOSE_TOTALS: partial geometry proxyVertices');

      // A package rewritten after its set was published makes the set stale.
      const stale = await packages(root, 'stale', 7);
      const staleSet = await assetSet(root, stale);
      await writeFile(join(root, 'stale-uv.json'), (await readFile(join(root, 'stale-uv.json'), 'utf8')) + ' ');
      fails(['--compose', 'stale=' + staleSet, ...tail], 'ASSET_SET_COMPOSE_STALE_PACKAGE: stale uvAudit');

      // A hand-edited set whose texture package belongs to another source (manifest hash updated to match).
      const mixed = JSON.parse(await readFile(ivySet, 'utf8'));
      mixed.packages.textures.root = mixed.roots.textureGpuRoot = mixed.roots.textureReferenceRoot = '/artifacts/main-textures/';
      mixed.packages.textures.manifestSha256 = sha256(await readFile(join(root, 'main-textures', 'manifest.json')));
      await mkdir(join(root, 'mixed-set'));
      await writeFile(join(root, 'mixed-set', 'asset-set.json'), JSON.stringify(mixed));
      await writeFile(join(root, 'mixed-set', 'manifest.json'), JSON.stringify(mixed));
      await copyFile(join(root, 'inputs', 'credits_license.txt'), join(root, 'mixed-set', 'credits_license.txt'));
      fails(['--compose', 'mixed=' + join(root, 'mixed-set', 'asset-set.json'), ...tail], 'ASSET_SET_COMPOSE_SOURCE_MISMATCH: mixed textures');
      // Without its completion marker (or with a different one) a set is incomplete.
      await writeFile(join(root, 'mixed-set', 'manifest.json'), '{}');
      fails(['--compose', 'mixed=' + join(root, 'mixed-set', 'asset-set.json'), ...tail], 'ASSET_SET_COMPOSE_INCOMPLETE: mixed');

      // Compositions do not nest, and every component must live under the artifacts root.
      succeeds([...both, ...tail.slice(0, -1), join(root, 'composed')]);
      fails(['--compose', 'nested=' + join(root, 'composed', 'asset-set.json'), ...tail], 'ASSET_SET_COMPOSE_COMPONENT_FORMAT: nested');
      const outside = await mkdtemp(join(resolve('.'), '.write-asset-set-compose-outside-'));
      try {
        await copyFile(mainSet, join(outside, 'asset-set.json'));
        fails(['--compose', 'main=' + join(outside, 'asset-set.json'), ...tail], 'ASSET_SET_OUTSIDE_ARTIFACTS');
      } finally { await rm(outside, { recursive: true, force: true }); }

      // Nothing was published by a failing run and no temporary directory is left behind.
      expect((await readdir(root)).filter(entry => entry.startsWith('.') || entry === 'out')).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }, 60_000);
});
