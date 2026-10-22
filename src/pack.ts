import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { encodeModel, type EncodeOptions } from './format.js';

export interface PackOptions extends EncodeOptions {
  /** Keep only the leading dimensions (Model2Vec's are PCA components, so they hold the most variance). */
  dims?: number;
}

/**
 * Packs a Model2Vec model folder (model.safetensors and tokenizer.json, as on Hugging Face, e.g.
 * minishlab/potion-base-8M) into the file `StaticModel.load` reads.
 */
export async function packModel(dir: string, { dims, ...encode }: PackOptions = {}): Promise<Uint8Array> {
  const st = await readFile(join(dir, 'model.safetensors'));
  const headerLen = Number(st.readBigUInt64LE(0));
  const header = JSON.parse(st.subarray(8, 8 + headerLen).toString());
  const others = Object.keys(header).filter((k) => k !== 'embeddings' && k !== '__metadata__');
  if (others.length) throw new Error(`unsupported tensors: ${others.join(', ')} (token weights or mappings)`);
  const info = header.embeddings;
  if (!info || info.dtype !== 'F32') throw new Error('expected an F32 "embeddings" tensor');
  const [n, full] = info.shape as [number, number];
  const keep = Math.min(dims ?? full, full);
  const data = st.subarray(8 + headerLen + info.data_offsets[0], 8 + headerLen + info.data_offsets[1]);
  const src = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));

  const tok = JSON.parse(await readFile(join(dir, 'tokenizer.json'), 'utf8'));
  if (tok.model?.type !== 'WordPiece' || tok.normalizer?.type !== 'BertNormalizer' || tok.normalizer.lowercase === false) {
    throw new Error('only lower-casing BERT (WordPiece) tokenizers are supported');
  }
  const tokens: string[] = [];
  for (const [t, id] of Object.entries(tok.model.vocab as Record<string, number>)) tokens[id] = t;
  for (const a of tok.added_tokens ?? []) tokens[a.id] = a.content;
  if (tokens.length !== n || tokens.includes(undefined as unknown as string)) throw new Error('vocabulary does not match the table');

  const vectors = new Float32Array(n * keep);
  for (let i = 0; i < n; i++) vectors.set(src.subarray(i * full, i * full + keep), i * keep);
  return encodeModel({ tokens, dims: keep, vectors }, encode);
}
