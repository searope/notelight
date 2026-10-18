// Type-checked (never run) by scripts/smoke_npm.py with `tsc --noEmit`, to
// check that the package's `types` export resolves under Node's module
// resolution and describes the API the README shows.
import init, {
  type Bm25Hit,
  type Chunk,
  type Hit,
  type TextChunk,
  type VectorHit,
  chunkText,
  Index,
  IndexWriter,
  version,
} from "notelight-wasm";

await init();
const v: string = version();

const empty = new Uint8Array();
const index = new Index(empty, empty, empty, null);
const bm25: Bm25Hit[] = index.searchBm25("query", 10);
const nearest: VectorHit[] = index.searchVector(new Float32Array(index.dim), 1);
const hybrid: Hit[] = index.search("query", new Float32Array(index.dim), 10);
const rank: number | null = hybrid[0].bm25Rank;
const chunk: Chunk = index.chunk(bm25[0].id);
const embedder: string | undefined = index.embedder;

const pieces: TextChunk[] = chunkText("# A\n\ntext\n", 1200, true);
const writer = new IndexWriter(0, null, null);
writer.add("a.md", pieces[0].headingPath, 1, 3, pieces[0].text, null);
const gen = writer.build(1);
const vectors: Uint8Array | undefined = gen.vectors;
new Index(gen.manifest, gen.docs, gen.bm25, vectors ?? null);

export { v, rank, chunk, embedder };
