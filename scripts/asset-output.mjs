import { constants } from 'node:fs';
import { lstat, mkdir, readdir, copyFile, link, unlink, rmdir, rm } from 'node:fs/promises';
import { basename, dirname, resolve, join } from 'node:path';

export async function exists(path) {
  try { await lstat(path); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

/** Delete only a builder-owned sibling staging directory, never the published output. */
export async function removeOwnTemporary(temporary, targetParent, output) {
  if (!(await exists(temporary))) return;
  const absolute = resolve(temporary);
  if (dirname(absolute) !== resolve(targetParent)
    || !basename(absolute).startsWith(`.${basename(output)}.tmp-`)) {
    throw new Error('refusing to remove a directory outside this build temporary root');
  }
  await rm(absolute, { recursive: true, force: true });
}

/** Claim the destination exclusively. The manifest is the completion marker;
 * it is linked last, since portable Node APIs lack no-replace directory rename.
 * Readers must require the manifest, not just the presence of a directory. */
export async function publishDirectory(temporary, output) {
  const source = resolve(temporary), destination = resolve(output);
  const directories = [], files = [];
  const visit = async (folder, prefix = '') => {
    for (const entry of await readdir(folder, { withFileTypes: true })) {
      const name = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) { directories.push(name); await visit(join(folder, entry.name), name); }
      else if (entry.isFile()) files.push(name);
      else throw new Error('build staging directory contains a non-regular entry');
    }
  };
  await visit(source);
  if (!files.includes('manifest.json')) throw new Error('build manifest is missing');
  await mkdir(destination); // EEXIST also rejects an empty directory created concurrently.
  const createdDirectories = [destination], createdFiles = [];
  try {
    for (const name of directories) {
      const path = join(destination, name);
      await mkdir(path); createdDirectories.push(path);
    }
    for (const name of [...files.filter(name => name !== 'manifest.json'), 'manifest.json']) {
      const path = join(destination, name);
      if (name === 'manifest.json') await link(join(source, name), path);
      else await copyFile(join(source, name), path, constants.COPYFILE_EXCL);
      createdFiles.push(path);
    }
  } catch (error) {
    // Only remove successfully created files; never recursively erase the
    // destination, which could contain another writer's unrecognized files.
    for (const path of createdFiles.reverse()) await unlink(path).catch(() => {});
    for (const path of createdDirectories.reverse()) await rmdir(path).catch(() => {});
    throw error;
  }
  await removeOwnTemporary(source, dirname(destination), destination);
}
