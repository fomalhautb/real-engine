#!/usr/bin/env node
// Runs one command against a private Vite dev server for this checkout and
// always closes the server afterwards, so capture scripts never attach to a
// server started from another worktree.
//   node scripts/with-dev-server.mjs [--port 5208] -- node scripts/capture-sponza-gi.mjs ...
// The port defaults to PLAYWRIGHT_PORT (else 5173) and must be free; the child
// receives SPONZA_BASE_URL and PLAYWRIGHT_PORT for that server. File watching
// and HMR are off: a capture never reloads because a source changed under it.
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2), separator = argv.indexOf('--');
if (separator < 0 || separator === argv.length - 1) throw Error('usage: with-dev-server [--port N] -- command [args...]');
const own = argv.slice(0, separator), command = argv.slice(separator + 1);
if (own.length && (own.length !== 2 || own[0] !== '--port')) throw Error('WITH_DEV_SERVER_ARGUMENTS');
const port = Number(own[1] ?? process.env.PLAYWRIGHT_PORT ?? 5173);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error('WITH_DEV_SERVER_PORT');

const server = await createServer({ root: ROOT, server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null } });
await server.listen();
const baseUrl = `http://127.0.0.1:${port}`;
process.stderr.write(`with-dev-server: ${baseUrl} (root ${ROOT})\n`);
let code = 1;
try {
  code = await new Promise((resolveExit, reject) => {
    const child = spawn(command[0], command.slice(1), { cwd: ROOT, stdio: 'inherit',
      env: { ...process.env, SPONZA_BASE_URL: baseUrl, PLAYWRIGHT_PORT: String(port) } });
    const forward = signal => child.kill(signal);
    process.once('SIGINT', forward); process.once('SIGTERM', forward);
    child.once('error', reject);
    child.once('exit', (exitCode, signal) => resolveExit(exitCode ?? (signal ? 1 : 0)));
  });
} finally { await server.close(); }
process.exitCode = code;
