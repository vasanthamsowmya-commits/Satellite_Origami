"""
rag_pipeline.py  —  CLOUD READY VERSION (DeepSeek API)
=======================================================
Retrieval  : BM25 + Topic Exact Match Boost
Generation : DeepSeek API

SETUP:
------
Local testing:
  set DEEPSEEK_API_KEY = "sk-xxxx"  directly below

Render cloud deployment:
  set DEEPSEEK_API_KEY in Render dashboard → Environment Variables
  code reads it automatically via os.environ
  never hardcode key in code when pushing to GitHub

Install:
  pip install requests flask flask-cors torch matplotlib numpy
"""

import json, math, re, os, requests

KNOWLEDGE_PATH = "knowledge_base.json"
TOP_K          = 4
K1             = 1.2
B              = 0.5

# ══════════════════════════════════════════════════════════════════
# DEEPSEEK API SETTINGS
# ── For LOCAL testing: paste your key directly below
# ── For RENDER cloud:  leave as "" and set in Render dashboard
# ══════════════════════════════════════════════════════════════════
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")   # reads from Render env
DEEPSEEK_MODEL   = "deepseek-chat"
DEEPSEEK_URL     = "https://api.deepseek.com/v1/chat/completions"

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in origami engineering for satellite structures. "
    "Answer using ONLY the context provided below. "
    "Do not use any outside knowledge. "
    "If the context does not fully answer the question, say so clearly. "
    "Be precise, technical, and concise. "
    "Keep answers under 100 words unless more detail is needed."
)

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "that","this","it","its","as","can","will","also","used","use","uses",
    "using","which","they","their","than","more","most","other","these",
    "those","such","when","where","how","what","who","each","both","all",
    "any","into","not","no","so","if","up","do","did","been","being","just",
    "very","some","about","there","here","then","only","even","still","tell",
    "give","show","explain","describe","define"
}

QUERY_EXPAND = {
    "satellite":     "satellite spacecraft orbit types overview",
    "satellites":    "satellite spacecraft orbit types overview",
    "origami":       "origami fold pattern crease deployment",
    "fold":          "origami fold pattern crease mountain valley",
    "solar":         "solar panel array deployment satellite power",
    "antenna":       "antenna deployable reflector satellite communication",
    "cubesat":       "cubesat small satellite deployable nanosatellite",
    "gps":           "navigation satellite GPS GNSS orbit positioning",
    "weather":       "weather satellite meteorological atmospheric monitoring",
    "communication": "communication satellite relay geostationary antenna",
    "telescope":     "space telescope deployable mirror aperture optics",
    "materials":     "materials kapton carbon fiber shape memory origami",
    "deployment":    "deployment mechanism fold unfold satellite structure",
    "miura":         "miura-ori fold pattern satellite solar panel deployment",
    "cvae":          "CVAE ETH Zurich model origami crease pattern generation",
    "eth":           "ETH Zurich CVAE dataset origami satellite",
    "kresling":      "kresling pattern cylindrical deployable bistable",
    "truss":         "truss structure deployable satellite boom",
    "reflector":     "reflector dish antenna satellite deployable",
    "flasher":       "flasher fold pattern rotational circular solar array",
    "yoshimura":     "yoshimura buckling pattern cylindrical deployable boom",
    "waterbomb":     "waterbomb base bistable fold satellite",
    "ikaros":        "IKAROS solar sail JAXA deployment flasher fold",
    "jwst":          "James Webb Space Telescope sunshield origami deployment",
    "iss":           "space station ISS solar array deployable accordion",
    "kapton":        "kapton polyimide film material satellite origami",
    "sma":           "shape memory alloy hinge deployment satellite",
    "gnn":           "graph neural network crease pattern classification",
    "rag":           "retrieval augmented generation knowledge base satellite",
}


# ══════════════════════════════════════════════════════════════════
# TOKENISER
# ══════════════════════════════════════════════════════════════════

def tokenize(text):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def expand_query(query):
    tokens   = re.findall(r'\b[a-z]+\b', query.lower())
    expanded = query
    for token in tokens:
        if token in QUERY_EXPAND:
            expanded = expanded + " " + QUERY_EXPAND[token]
    return expanded

def topic_similarity(query_tokens, topic):
    topic_tokens  = set(tokenize(topic))
    query_set     = set(query_tokens)
    if not topic_tokens:
        return 0.0
    overlap        = len(topic_tokens & query_set)
    full_match     = 1.0 if topic_tokens.issubset(query_set) else 0.0
    query_coverage = overlap / max(len(query_set), 1)
    return overlap * 3.0 + full_match * 10.0 + query_coverage * 5.0


# ══════════════════════════════════════════════════════════════════
# BM25 INDEX
# ══════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, documents):
        self.docs   = documents
        self.topics = [d.get("topic", "") for d in documents]
        self.N      = len(documents)
        self._build()

    def _build(self):
        self.tokenized = []
        for doc in self.docs:
            topic_tokens = tokenize(doc.get("topic", "")) * 3
            text_tokens  = tokenize(doc["text"])
            self.tokenized.append(topic_tokens + text_tokens)

        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.N, 1)

        self.df = {}
        for tokens in self.tokenized:
            for word in set(tokens):
                self.df[word] = self.df.get(word, 0) + 1

        self.vectors = []
        for tokens in self.tokenized:
            tf  = {}
            for w in tokens:
                tf[w] = tf.get(w, 0) + 1
            dl  = len(tokens)
            vec = {}
            for word, cnt in tf.items():
                tf_bm25 = (cnt * (K1 + 1)) / (
                    cnt + K1 * (1 - B + B * dl / self.avgdl)
                )
                idf = math.log(
                    (self.N - self.df[word] + 0.5) /
                    (self.df[word] + 0.5) + 1
                )
                vec[word] = tf_bm25 * idf
            self.vectors.append(vec)

    def search(self, query, top_k=TOP_K):
        expanded   = expand_query(query)
        qtoks      = tokenize(expanded)
        orig_qtoks = tokenize(query)
        if not qtoks:
            return []

        scores = []
        for i, vec in enumerate(self.vectors):
            bm25_score  = sum(vec.get(w, 0) for w in qtoks)
            topic_boost = topic_similarity(orig_qtoks, self.topics[i])
            scores.append((bm25_score + topic_boost, i))

        scores.sort(reverse=True)
        results = []
        for score, i in scores[:top_k]:
            if score < 0.01:
                break
            results.append({
                "text":  self.docs[i]["text"],
                "topic": self.docs[i].get("topic", ""),
                "id":    self.docs[i].get("id", ""),
                "score": round(score, 2)
            })
        return results


# ══════════════════════════════════════════════════════════════════
# DEEPSEEK ANSWERER
# ══════════════════════════════════════════════════════════════════

def ask_deepseek(question, chunks):
    """
    Sends retrieved chunks as context to DeepSeek API.
    DeepSeek reads the context and writes a real intelligent answer.
    """
    if not DEEPSEEK_API_KEY:
        print("[DeepSeek] ERROR: No API key found.")
        print("[DeepSeek] Set DEEPSEEK_API_KEY in Render dashboard")
        print("[DeepSeek] or paste directly: DEEPSEEK_API_KEY = 'sk-xxx'")
        return None

    context = "\n\n".join(
        f"[Source {i+1} — {c['topic']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    user_message = (
        f"### CONTEXT\n{context}\n\n"
        f"### QUESTION\n{question}\n\n"
        f"### ANSWER"
    )

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type":  "application/json"
            },
            json={
                "model":    DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                "max_tokens":  300,
                "temperature": 0.3
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"[DeepSeek] Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[DeepSeek] Request failed: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════

class RAGPipeline:

    def __init__(self, knowledge_path=KNOWLEDGE_PATH):
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(
                f"knowledge_base.json not found.\n"
                f"Run from the satellite_final folder."
            )
        with open(knowledge_path, encoding="utf-8") as f:
            self.docs = json.load(f)

        self.index = BM25Index(self.docs)

        # Check API key on startup
        if DEEPSEEK_API_KEY:
            print(f"[RAG] Ready — {len(self.docs)} documents indexed")
            print(f"[RAG] Generator : DeepSeek API ({DEEPSEEK_MODEL})")
            print(f"[RAG] API Key   : {DEEPSEEK_API_KEY[:8]}••••••••••••")
        else:
            print(f"[RAG] Ready — {len(self.docs)} documents indexed")
            print(f"[RAG] WARNING: No DeepSeek API key found!")
            print(f"[RAG] Set DEEPSEEK_API_KEY in Render environment variables")

    def retrieve(self, query, top_k=TOP_K):
        return self.index.search(query, top_k=top_k)

    def answer(self, question):
        """Returns (answer_text, sources_list)"""
        chunks = self.retrieve(question)

        if not chunks:
            return "I don't have information about that in my knowledge base.", []

        # Ask DeepSeek
        response = ask_deepseek(question, chunks)

        if response:
            return response, chunks

        # Fallback — return best chunk text if API fails
        best = chunks[0]
        return f"[{best['topic']}]\n\n{best['text']}", chunks


# ══════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════

def run_test(rag):
    print()
    print("=" * 65)
    print("  RETRIEVAL TEST  (BM25 only — no API calls)")
    print("=" * 65)

    tests = [
        ("Flasher fold pattern",            "Flasher fold"),
        ("Yoshimura pattern",               "Yoshimura"),
        ("Waterbomb base",                  "Waterbomb"),
        ("Kresling pattern",                "Kresling"),
        ("Miura-ori fold",                  "Miura-ori"),
        ("Star fold pattern",               "Star fold"),
        ("what is satellite",               "satellite"),
        ("what is origami",                 "origami"),
        ("ETH Zurich CVAE model",           "CVAE"),
        ("JWST sunshield origami",          "Webb"),
        ("IKAROS solar sail",               "IKAROS"),
        ("shape memory alloy hinge",        "memory alloy"),
        ("CubeSat deployable solar panel",  "CubeSat"),
        ("communication satellite antenna", "communication"),
    ]

    passed = 0
    for question, must_contain in tests:
        results = rag.retrieve(question, top_k=1)
        if results:
            topic  = results[0]["topic"]
            ok     = must_contain.lower() in topic.lower()
            if ok: passed += 1
            status = "PASS ✓" if ok else "FAIL ✗"
            print(f"  {status}  [{results[0]['score']:5.2f}]  {topic}")
            print(f"           Q: {question}")
        else:
            print(f"  FAIL ✗  No result  Q: {question}")
        print()

    print(f"  Result: {passed}/{len(tests)} correct")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 65)
    print("  ORIGAMI SATELLITE RAG  —  DeepSeek Cloud Version")
    print("  Retrieval  : BM25 + Topic Exact Match Boost")
    print("  Generator  : DeepSeek API")
    print("=" * 65)

    rag = RAGPipeline()
    run_test(rag)

    print("Type your question. Commands: 'sources'  'exit'\n")
    last_sources = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "sources":
            if last_sources:
                print("\nSources:")
                for s in last_sources:
                    print(f"  [{s['score']}]  {s['topic']}")
            continue

        answer, last_sources = rag.answer(q)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
