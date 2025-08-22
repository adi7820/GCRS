import json, re
import pandas as pd
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Any, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
import graphistry
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import process, fuzz
import time

# graphistry.register(
#     api=3,
#     server="hub.graphistry.com",
#     username="Random2678",
#     password="Adi@7820915"
# )

load_dotenv(override=True)

# Stage 1: load product catalogue and chat logs
def load_product_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"product_id","product_name","category","brand","price","description"}
    if missing := required - set(df.columns):
        raise ValueError(f"catalog missing {missing}")
    return df

def load_conversations(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    required = {"conversation_id","speaker","message"}
    if missing := required - set(df.columns):
        raise ValueError(f"log missing {missing}")
    return df

# Stage 2: triple extraction (stub)
DEFAULT_TRIPLE_PROMPT = """You are an information extraction assistant.
Extract factual triples of the form [subject, subject_type, relation, object, object_type] from the text.
Only use relations relevant to product recommendation: belongs_to, manufactured_by, has_feature, compatible_with, similar_to.
Entity names must be <=4 words. Output a JSON array of arrays.
Description: {input}
""".strip()

_TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": [{"type":"string"} for _ in range(5)]
            }
        }
    },
    "required": ["triples"],
    "additionalProperties": False
}

ALLOWED_RELATIONS = {
    "belongs_to", "manufactured_by", "has_feature", "compatible_with", "similar_to"
}

def _coerce_triples(obj) -> List[Tuple[str,str,str,str,str]]:
    # Accept either array-only or {"triples": [...]} shapes
    triples_raw = obj if isinstance(obj, list) else obj.get("triples", [])
    clean = []
    for t in triples_raw:
        if not isinstance(t, (list, tuple)) or len(t) != 5:
            continue
        s, st, r, o, ot = [str(x).strip() for x in t]
        # minimal sanity + optional relation whitelist
        if not (s and r and o):
            continue
        if ALLOWED_RELATIONS and r not in ALLOWED_RELATIONS:
            continue
        clean.append((s, st, r, o, ot))
    # dedupe while preserving order
    return list(dict.fromkeys(clean))

def _fallback_extract_first_json_array(text: str):
    # Upgrade: capture the first {...} OR [...] block (not only arrays)
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        print("Extracted JSON:", data)
        return _coerce_triples(data)
    except Exception:
        return []

def call_llm_for_triples(
    text: str,
    llm_api_key: str,
    prompt: str = DEFAULT_TRIPLE_PROMPT
) -> List[Tuple[str, str, str, str, str]]:
    """
    Returns list of (subject, subject_type, relation, object, object_type)
    """
    client = OpenAI(api_key=llm_api_key)
    final_prompt = (
        "Return ONLY JSON. No prose.\n"
        "Schema: {\"triples\": [[\"subject\",\"subject_type\",\"relation\",\"object\",\"object_type\"], ...]}\n\n"
        + prompt.format(input=text)
    )

    try:
        # Prefer structured output
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract product knowledge as JSON only."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={  # enforce JSON
                "type": "json_schema",
                "json_schema": {
                    "name": "triple_list",
                    "schema": _TRIPLE_SCHEMA,
                    "strict": True
                }
            }
        )
        content = resp.choices[0].message.content
        data = json.loads(content)               # guaranteed JSON if model honors response_format
        triples = _coerce_triples(data)
        if triples:
            return triples
        # fall through to regex if empty
        return _fallback_extract_first_json_array(content)

    except Exception as e:
        # If the model ignored response_format or an SDK/network hiccup occurred,
        # try a plain call + fallback JSON extraction.
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Respond with JSON only; no explanations."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.1,
                max_tokens=512,
            )
            content = resp.choices[0].message.content
            # try strict first
            try:
                data = json.loads(content)
                triples = _coerce_triples(data)
                if triples:
                    return triples
            except Exception:
                pass
            # fallback: extract first JSON array in mixed text
            return _fallback_extract_first_json_array(content)
        except Exception as e2:
            print(f"Triple extraction failed: {e2}")
            return []
def extract_triples_from_catalog(products_df: pd.DataFrame, llm_api_key: Any = None) -> List[Tuple[str,str,str,str,str]]:
    triples = []
    for desc in products_df['description']:
        triples.extend(call_llm_for_triples(desc, llm_api_key))
    return triples

# Stage 3: build graph

def coerce_triples(triples_raw):
    # Handle {"triples": [...]} wrapper
    if isinstance(triples_raw, dict) and "triples" in triples_raw:
        triples_raw = triples_raw["triples"]

    clean = []
    bad = []

    for t in triples_raw:
        # Case 1: list/tuple
        if isinstance(t, (list, tuple)):
            if len(t) >= 5:
                clean.append([t[0], t[1], t[2], t[3], t[4]])  # trim extras
            else:
                bad.append(("too_short", t))

        # Case 2: dict with keys
        elif isinstance(t, dict):
            # try common keys
            keys = ["subject","subject_type","relation","object","object_type"]
            if all(k in t for k in keys):
                clean.append([t[k] for k in keys])
            else:
                bad.append(("dict_missing_keys", t))

        # Case 3: string (maybe JSON-encoded)
        elif isinstance(t, str):
            try:
                tj = json.loads(t)
                if isinstance(tj, (list, tuple)) and len(tj) >= 5:
                    clean.append([tj[0], tj[1], tj[2], tj[3], tj[4]])
                elif isinstance(tj, dict):
                    if all(k in tj for k in keys):
                        clean.append([tj[k] for k in keys])
                    else:
                        bad.append(("str_json_missing_keys", t))
                else:
                    bad.append(("str_bad_shape", t))
            except Exception:
                # maybe CSV-ish: "s,st,r,o,ot,extra"
                parts = [p.strip() for p in t.split(",")]
                if len(parts) >= 5:
                    clean.append(parts[:5])
                else:
                    bad.append(("str_not_json_or_csv", t))
        else:
            bad.append(("unknown_type", t))

    if bad:
        # optional: log a quick summary once
        print(f"[coerce_triples] Skipped {len(bad)} malformed items. "
              f"Example issue: {bad[0][0]} -> {bad[0][1]}")
    return clean
@dataclass
class IdMaps:
    entity_to_id: Dict[str,int]
    id_to_entity: Dict[int,str]
    relation_to_id: Dict[str,int]
    id_to_relation: Dict[int,str]

def build_graph_from_triples(triples: Iterable[Tuple[str,str,str,str,str]]) -> Tuple[nx.DiGraph, IdMaps]:
    G = nx.DiGraph()
    ent2id, rel2id = {}, {}
    next_e = next_r = 0
    for s, st, r, o, ot in triples:
        if s not in ent2id: ent2id[s] = next_e; next_e += 1
        if o not in ent2id: ent2id[o] = next_e; next_e += 1
        if r not in rel2id: rel2id[r] = next_r; next_r += 1
        G.add_node(ent2id[s], name=s, type=st)
        G.add_node(ent2id[o], name=o, type=ot)
        G.add_edge(ent2id[s], ent2id[o], relation=r)
    id_maps = IdMaps(ent2id, {v:k for k,v in ent2id.items()}, rel2id, {v:k for k,v in rel2id.items()})
    return G, id_maps

def save_graph_and_lexicon(
    graph: nx.DiGraph,
    id_maps: IdMaps,
    lexicon: Tuple[Dict[str, str], Dict[str, int]],
    cache_dir: str = "cache",
) -> None:
    """
    Persist the graph, ID mappings and normalised lexicon to disk.

    * ``graph`` is serialised to GraphML.
    * ``id_maps`` is saved as JSON containing all four mappings.
    * ``lexicon`` is saved as JSON with two keys: ``norm2orig`` and ``norm2id``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Save graph
    graph_path = os.path.join(cache_dir, "graph.graphml")
    nx.write_graphml(graph, graph_path)
    # Save id_maps
    id_maps_path = os.path.join(cache_dir, "id_maps.json")
    with open(id_maps_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_to_id": id_maps.entity_to_id,
                "id_to_entity": id_maps.id_to_entity,
                "relation_to_id": id_maps.relation_to_id,
                "id_to_relation": id_maps.id_to_relation,
            },
            f,
        )
    # Save lexicon
    lexicon_path = os.path.join(cache_dir, "lexicon.json")
    norm2orig, norm2id = lexicon
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump({"norm2orig": norm2orig, "norm2id": norm2id}, f)

# Stage 4: vector index (simple FAISS example)

def normalize(text: str) -> str:
    # lower, collapse spaces, strip punctuation
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_normalized_lexicon(id_maps) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Returns:
      norm_name -> original_name
      norm_name -> node_id
    """
    norm2orig, norm2id = {}, {}
    for name, nid in id_maps.entity_to_id.items():
        n = normalize(name)
        norm2orig[n] = name
        norm2id[n] = nid
    return norm2orig, norm2id

def ngrams(words: List[str], max_n: int = 4):
    for n in range(max_n, 0, -1):  # longest first
        for i in range(0, len(words) - n + 1):
            yield " ".join(words[i:i+n])

def extract_seed_entities(query: str,
                          id_maps,
                          max_ngram: int = 4,
                          use_fuzzy: bool = True,
                          fuzzy_threshold: int = 88) -> List[str]:
    """
    Returns a list of entity names (original casing) found in the query
    by n-gram exact match + optional fuzzy fallback.
    """
    norm2orig, _ = build_normalized_lexicon(id_maps)
    tokens = normalize(query).split()

    # 1) Exact n-gram matches against normalized lexicon
    hits = []
    seen = set()
    for span in ngrams(tokens, max_ngram):
        nspan = normalize(span)
        if nspan in norm2orig and nspan not in seen:
            hits.append(norm2orig[nspan])
            seen.add(nspan)

    if hits:
        return hits

    # 2) Optional fuzzy fallback (best match over the whole query)

    candidates = list(norm2orig.keys())
    best, score, _ = process.extractOne(
        normalize(query), candidates, scorer=fuzz.token_set_ratio
    )
    if score >= fuzzy_threshold:
        return [norm2orig[best]]

    return []

def load_graph_and_lexicon(cache_dir: str = "cache") -> Optional[Tuple[nx.DiGraph, IdMaps, Tuple[Dict[str, str], Dict[str, int]]]]:
    """
    Load graph, ID mappings and lexicon from disk if they exist.  Returns
    ``None`` if any component is missing.  ID maps are coerced back to
    correct types (int keys where appropriate).
    """
    graph_path = os.path.join(cache_dir, "graph.graphml")
    id_maps_path = os.path.join(cache_dir, "id_maps.json")
    lexicon_path = os.path.join(cache_dir, "lexicon.json")
    if not (os.path.exists(graph_path) and os.path.exists(id_maps_path) and os.path.exists(lexicon_path)):
        return None
    # Read graph
    G = nx.read_graphml(graph_path)
    # GraphML loads node IDs as strings; convert to int
    G_int = nx.DiGraph()
    for u, attrs in G.nodes(data=True):
        G_int.add_node(int(u), **attrs)
    for u, v, attrs in G.edges(data=True):
        G_int.add_edge(int(u), int(v), **attrs)
    # Load id_maps
    with open(id_maps_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert string keys back to ints where necessary
    entity_to_id = {k: int(v) for k, v in data["entity_to_id"].items()}
    id_to_entity = {int(k): v for k, v in data["id_to_entity"].items()}
    relation_to_id = {k: int(v) for k, v in data["relation_to_id"].items()}
    id_to_relation = {int(k): v for k, v in data["id_to_relation"].items()}
    id_maps = IdMaps(entity_to_id, id_to_entity, relation_to_id, id_to_relation)
    # Load lexicon
    with open(lexicon_path, "r", encoding="utf-8") as f:
        lex = json.load(f)
    norm2orig: Dict[str, str] = lex.get("norm2orig", {})
    norm2id: Dict[str, int] = {k: int(v) for k, v in lex.get("norm2id", {}).items()}
    return G_int, id_maps, (norm2orig, norm2id)

class SimpleHybridVectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def _norm(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def build_index(self, texts: List[str], metas: List[Dict[str,Any]]):
        vecs = self._norm(self.model.encode(texts, show_progress_bar=False).astype('float32'))
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.embeddings = vecs
        self.metadata = metas

    def search(self, query: str, top_k: int=5) -> List[Dict[str,Any]]:
        qvec = self._norm(self.model.encode([query]).astype('float32'))
        scores, indices = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results
    
def index_documents(products_df: pd.DataFrame, conv_df: pd.DataFrame) -> SimpleHybridVectorIndex:
    index = SimpleHybridVectorIndex()
    texts, metas = [], []
    for _, row in products_df.iterrows():
        texts.append(row['description'])
        metas.append({"type":"product","product_id":row['product_id'],"source":"catalog"})
    for _, row in conv_df.iterrows():
        texts.append(row['message'])
        metas.append({"type":"conversation","conversation_id":row['conversation_id'],"source":"chat"})
    index.build_index(texts, metas)
    return index

def save_vector_index(index: SimpleHybridVectorIndex, cache_dir: str = "cache") -> None:
    """
    Persist the FAISS index and associated metadata to disk.  The
    embeddings themselves are not saved because queries can be embedded on
    demand.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if index.index is None:
        raise ValueError("Index has not been built.")
    faiss_path = os.path.join(cache_dir, "faiss.index")
    faiss.write_index(index.index, faiss_path)
    meta_path = os.path.join(cache_dir, "index_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(index.metadata, f)
        
def load_vector_index(cache_dir: str = "cache", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[SimpleHybridVectorIndex]:
    """
    Load the FAISS index and metadata from disk if they exist.  Returns
    ``None`` if either component is missing.
    """
    faiss_path = os.path.join(cache_dir, "faiss.index")
    meta_path = os.path.join(cache_dir, "index_metadata.json")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        return None
    idx = SimpleHybridVectorIndex(model_name)
    idx.index = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        idx.metadata = json.load(f)
    return idx

# Stage 5: personalised PageRank
def personalised_pagerank(graph: nx.DiGraph, id_maps: IdMaps, seed_entities: Iterable[str], alpha: float=0.85, top_k: int=10):
    seed_ids = [id_maps.entity_to_id[e] for e in seed_entities if e in id_maps.entity_to_id]
    if not seed_ids: return []
    personalization = {n:0.0 for n in graph.nodes()}
    for sid in seed_ids: personalization[sid] = 1.0/len(seed_ids)
    pr = nx.pagerank(graph, alpha=alpha, personalization=personalization)
    ranked = [(id_maps.id_to_entity[n], score) for n, score in pr.items() if n not in seed_ids]
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_k]

def retrieve_candidates_and_examples(query, graph, id_maps, conv_df, index, top_n=3):
    # ignore `graph` and `id_maps` here
    vect_res = index.search(query, top_k=50)
    conv_scores = {}
    for item in vect_res:
        if item["type"] == "conversation":
            cid = item["conversation_id"]
            conv_scores[cid] = conv_scores.get(cid, 0.0) + item["score"]
    top_convs = sorted(conv_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    example_convs = []
    for cid, _ in top_convs:
        msgs = conv_df[conv_df["conversation_id"] == cid].sort_values("timestamp")["message"].tolist()
        example_convs.append({"conversation_id": cid, "messages": msgs})
    return example_convs

# Stage 6: prompt construction and LLM call
def _normalize_candidates(candidates: List[Any]) -> List[Dict[str, Any]]:
    """
    Accepts:
      - [("PlayFun", 0.07), ("HomeEase", 0.06), ...]
      - [{"entity_name": "PlayFun", "score": 0.07}, ...]
      - Mixed and slightly messy strings like "'PlayFun'" or "('ultra Camera'"
    Returns:
      - [{"entity_name": "PlayFun", "score": 0.07}, ...] (deduped by name, max score)
    """
    norm = []
    for c in candidates:
        if isinstance(c, dict):
            name = c.get("entity_name") or c.get("name") or c.get("entity") or c.get("title")
            score = c.get("score", 0.0)
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            name, score = c[0], c[1]
        else:
            continue

        # coerce & clean name
        name = str(name)
        # strip leading/trailing quotes/parentheses and extra spaces
        name = re.sub(r"^[\\('\"\\s]+|[\\)'\"\\s]+$", "", name).strip()
        # optional: collapse multiple spaces
        name = re.sub(r"\\s+", " ", name)

        try:
            score = float(score)
        except Exception:
            score = 0.0

        if name:
            norm.append({"entity_name": name, "score": score})

    # dedupe by name → keep max score
    agg = {}
    for d in norm:
        if d["entity_name"] in agg:
            if d["score"] > agg[d["entity_name"]]:
                agg[d["entity_name"]] = d["score"]
        else:
            agg[d["entity_name"]] = d["score"]

    # sort by score desc
    return [{"entity_name": k, "score": v} for k, v in sorted(agg.items(), key=lambda x: x[1], reverse=True)]


def construct_prompt(query: str,
                     candidate_products: List[Any],
                     example_conversations: List[Dict[str, Any]],
                     system_msg: str = "You are a helpful recommendation assistant.") -> str:
    lines = [system_msg, "", f"User query: {query}", ""]

    # normalize candidates
    cands = _normalize_candidates(candidate_products)
    if cands:
        lines.append("Candidate products:")
        for i, item in enumerate(cands, 1):
            # defensive: missing keys won't crash
            nm = item.get("entity_name", "")
            sc = item.get("score", 0.0)
            lines.append(f"{i}. {nm} (score={sc:.3f})")
    else:
        lines.append("Candidate products: (none found)")

    # examples (defensive handling)
    if example_conversations:
        lines.append("")
        lines.append("Past conversations:")
        for ex in example_conversations:
            cid = ex.get("conversation_id", "N/A")
            msgs = ex.get("messages") or []
            lines.append(f"Conversation {cid}:")
            # only show first 5 lines to keep prompt concise
            for msg in msgs[:5]:
                lines.append(f"  - {msg}")
            lines.append("")
    else:
        lines.append("")
        lines.append("Past conversations: (none found)")

    lines.append("Based on the query, candidates and examples above, recommend the best products and explain why.")
    return "\n".join(lines)

def call_llm_for_response(prompt: str, llm_api_key: str) -> str:
    """
    Calls OpenAI's GPT-4o with the given prompt and returns the model's text output.
    Works with openai>=1.0.0
    """
    client = OpenAI(api_key=llm_api_key)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for product recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        # Extract the text content from the first choice
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return ""

if __name__ == "__main__":
    catalog = load_product_catalog("./synthetic_data/products.csv")
    # print(catalog['description'])
    conversations = load_conversations("./synthetic_data/conversations.jsonl")
    # print(conversations.head())
    # product_description = catalog['description'].iloc[1]
    # print("Product description:", product_description)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # triples = call_llm_for_triples(product_description, llm_api_key=OPENAI_API_KEY)
    # print("Extracted triples:", triples)
    # triplets_list = extract_triples_from_catalog(catalog, llm_api_key=OPENAI_API_KEY)
    # with open("triplets_list.txt", "w") as file:
    #     for item in triplets_list:
    #         file.write(f"{item}\n")
    with open("triplets_list.txt", "r") as file:
        triplets_list = [line.strip() for line in file]
        
    # print(triplets_list)
    triples5 = coerce_triples(triplets_list)
    # print(triples5)
    # G, id_maps = build_graph_from_triples(triples5)
    
    # print("Graph nodes:", G.number_of_nodes())
    # print("Graph edges:", G.number_of_edges())
    # print("Entity to ID mapping:", id_maps.entity_to_id)
    # print("Relation to ID mapping:", id_maps.relation_to_id)
    # print("ID to Entity mapping:", id_maps.id_to_entity)
    # print("ID to Relation mapping:", id_maps.id_to_relation)
    # print("Graph edges (triples):", list(G.edges(data=True)))
    # print("Graph nodes (entities):", list(G.nodes(data=True)))
    # print(G)
    # print(id_maps)
    # g_plot = graphistry.bind(source='src', destination='dst').from_networkx(G)
    # g_plot.plot()
    # norm2orig, norm2id = build_normalized_lexicon(id_maps)

    # save_graph_and_lexicon(G, id_maps, (norm2orig, norm2id))
    start_time = time.time()
    G, id_maps, lexicon = load_graph_and_lexicon()
    # print(time.time() - start_time, "seconds to load graph and lexicon")
    query = "Find a camera similar to Canon EOS R6 but cheaper"
    start_time = time.time()
    seed_entities = extract_seed_entities(query, id_maps, max_ngram=4, use_fuzzy=True)
    print("Seeds:", seed_entities)
    candidates = personalised_pagerank(G, id_maps, seed_entities, top_k=10)
    # print(time.time() - start_time, "seconds")
    # print(candidates)
    # query_entities = [tok.strip(' ,.!?').lower() for tok in query.split()]
    # print(query_entities)
    # known = [e for e in query_entities if e in id_maps.entity_to_id]
    # print(known)
    # result = personalised_pagerank(G, id_maps)
    # print("Personalised PageRank results:", result)

    # index = index_documents(catalog, conversations)
    # save_vector_index(index)
    index = load_vector_index()
    # print(index.search("What is the best smartphone under $500?"))
    # start_time = time.time()
    examples = retrieve_candidates_and_examples(query, G, id_maps, conversations, index)
    # # print(time.time() - start_time, "seconds")

    # print(candidates)
    # print(examples)

    prompt = construct_prompt(query, candidates, examples)
    # print(prompt)
    
    response = call_llm_for_response(prompt, llm_api_key=OPENAI_API_KEY)
    
    print("LLM response:", response)