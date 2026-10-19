import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

const { publishDirectory } = await import(pathToFileURL(resolve('scripts/asset-output.mjs')).href) as {
  publishDirectory: (temporary: string, output: string) => Promise<void>;
};

async function removeOwnTestRoot(root: string): Promise<void> {
  const absolute = resolve(root);
  if (dirname(absolute) !== resolve('.') || !basename(absolute).startsWith('.asset-output-test-')) {
    throw new Error('refusing to remove a directory outside this test root');
  }
  await rm(absolute, { recursive: true, force: true });
}

async function stage(root: string, output: string, id: string): Promise<string> {
  const temporary = await mkdtemp(join(root, `.${basename(output)}.tmp-`));
  await mkdir(join(temporary, 'nested'));
  await writeFile(join(temporary, 'nested', 'payload.bin'), Buffer.from([id.charCodeAt(0), 0, 255]));
  await writeFile(join(temporary, 'nested', 'label.txt'), `payload-${id}`);
  await writeFile(join(temporary, 'manifest.json'), `${JSON.stringify({ id, payload: `payload-${id}` })}\n`);
  return temporary;
}

describe('exclusive asset output publishing', () => {
  it('publishes exactly one complete staging tree under concurrent contention', async () => {
    const root = await mkdtemp(join(resolve('.'), '.asset-output-test-'));
    try {
      const output = join(root, 'published');
      const left = await stage(root, output, 'L');
      const right = await stage(root, output, 'R');
      const results = await Promise.allSettled([
        publishDirectory(left, output), publishDirectory(right, output),
      ]);
      expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
      expect(results.filter(result => result.status === 'rejected')).toHaveLength(1);
      const winner = results[0].status === 'fulfilled' ? 'L' : 'R';
      const loser = winner === 'L' ? right : left;
      expect((await readdir(output)).sort()).toEqual(['manifest.json', 'nested']);
      expect((await readdir(join(output, 'nested'))).sort()).toEqual(['label.txt', 'payload.bin']);
      expect(JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'))).toEqual({
        id: winner, payload: `payload-${winner}`,
      });
      expect(await readFile(join(output, 'nested', 'label.txt'), 'utf8')).toBe(`payload-${winner}`);
      expect(await readFile(join(output, 'nested', 'payload.bin')))
        .toEqual(Buffer.from([winner.charCodeAt(0), 0, 255]));
      expect(await readFile(join(loser, 'manifest.json'), 'utf8')).toContain(`"id":"${winner === 'L' ? 'R' : 'L'}"`);
    } finally {
      await removeOwnTestRoot(root);
    }
  });

  it('rejects a destination that already exists, even when empty', async () => {
    const root = await mkdtemp(join(resolve('.'), '.asset-output-test-'));
    try {
      const output = join(root, 'published');
      await mkdir(output);
      const temporary = await stage(root, output, 'A');
      await expect(publishDirectory(temporary, output)).rejects.toThrow();
      expect(await readdir(output)).toEqual([]);
      expect(await readFile(join(temporary, 'manifest.json'), 'utf8')).toContain('"id":"A"');
    } finally {
      await removeOwnTestRoot(root);
    }
  });
});
