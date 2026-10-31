// Worker thread for the offline BC7 encoder. Each task encodes a contiguous
// range of 4-pixel block rows of one level; results are pure functions of the
// pixels and options, so the pool's chunking never changes the output bytes.
import { parentPort } from 'node:worker_threads';
import { encodeBc7Rows } from '../src/texture-bc7.ts';

parentPort.on('message', task => {
  try {
    const pixels = new Uint8Array(task.pixels);
    const blocks = encodeBc7Rows(pixels, task.width, task.height, task.rowStart, task.rowEnd, task.options);
    parentPort.postMessage({ id: task.id, blocks }, [blocks.buffer]);
  } catch (error) {
    parentPort.postMessage({ id: task.id, error: error instanceof Error ? error.message : String(error) });
  }
});
