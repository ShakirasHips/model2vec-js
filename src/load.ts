// Fetching files: URLs everywhere, and local paths in Node (the Node entry point registers a file reader, so the
// browser build never imports node:fs).

/** Where to load a file from: a URL, a path (Node), or the bytes themselves. */
export type Source = string | URL | ArrayBuffer | ArrayBufferView;

type FileReader = (path: string | URL) => Promise<Uint8Array>;
let readFile: FileReader | null = null;

/** @internal Used by the Node entry point. */
export function setFileReader(reader: FileReader): void {
  readFile = reader;
}

/** The bundled model: potion-base-8M's first 128 dimensions, packed at 4 bits. */
export const DEFAULT_MODEL = new URL('../models/potion-base-8M-128d.bin', import.meta.url);

export async function loadBytes(src: Source): Promise<Uint8Array> {
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (ArrayBuffer.isView(src)) return new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
  const url = src instanceof URL ? src : null;
  if (readFile && (url ? url.protocol === 'file:' : !/^[a-z][a-z0-9+.-]*:\/\//i.test(src as string))) {
    return readFile(url ?? (src as string));
  }
  const res = await fetch(src as string | URL);
  if (!res.ok) throw new Error(`${String(src)}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}
