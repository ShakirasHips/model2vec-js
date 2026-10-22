// The Node entry point: everything in the browser one, plus loading from local paths and packing models.
import { readFile } from 'node:fs/promises';
import { setFileReader } from './load.js';

setFileReader(async (path) => new Uint8Array(await readFile(path)));

export * from './index.js';
export { packModel, type PackOptions } from './pack.js';
