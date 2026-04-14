#!/usr/bin/env python3
"""
EFSG Submission - Direct from working Notebook
Minimal changes for TIRA compatibility
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
import hashlib
import re
import threading
from typing import List, Optional, Dict, Any, Literal

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────
# COPY YOUR WORKING NOTEBOOK CODE HERE (sections 0-2)
# ─────────────────────────────────────────────────────────────

# CONFIG (from your notebook)
GENERATION_MODEL     = 'llama-3.3-70b-versatile'
EXTRACTION_MODEL     = 'llama-3.1-8b-instant'
EMBED_MODEL          = 'intfloat/multilingual-e5-large'
NLI_MODEL            = 'cross-encoder/nli-deberta-v3-small'
SENTENCES_TARGET     = 80
HYDE_N_HYPS          = 3
TOP_K_RETRIEVAL      = 20
FACT_POOL_RATIO      = 2.5
NLI_PATH_A           = 0.85
NLI_PATH_B           = 0.60
NLI_COHERENCE_FLOOR  = 0.60
RATE_LIMIT_RPM       = 25
MAX_RETRIES          = 5

# ─────────────────────────────────────────────────────────────
# SCHEMAS (from your notebook §1)
# ─────────────────────────────────────────────────────────────

class TopicJSON(BaseModel):
    topic_id:          str
    request_id:        Optional[str] = None
    collection_id:     Optional[str] = None
    title:             str
    background:        str
    problem_statement: str
    limit:             int
    preferred_languages: List[str] = Field(default_factory=lambda: ['en'])

class ReportIntent(BaseModel):
    topic_id:      str
    goal:          str
    language:      str
    section_hints: List[str]
    char_budget:   int

class SectionPlan(BaseModel):
    section_id:         int
    title:              str
    epistemic_contract: str
    word_budget:        int
    char_budget:        int
    sentence_budget:    int
    retrieval_queries:  List[str] = Field(default_factory=list)

class RetrievedDoc(BaseModel):
    doc_id:     str
    text:       str
    source:     str
    language:   str
    score:      float
    section_id: int

class AtomicFact(BaseModel):
    fact_id:       str
    text:          str
    source_doc_id: str
    section_id:    int
    corroboration: int           = 1
    cluster_id:    Optional[int] = None
    used:          bool          = False

class FactPool(BaseModel):
    topic_id:   str
    sealed:     bool = False
    facts:      List[AtomicFact]     = Field(default_factory=list)
    by_section: Dict[int, List[str]] = Field(default_factory=dict)
    _global_used: Dict[str, int] = PrivateAttr(default_factory=dict)

    def seal(self): 
        self.sealed = True

    def mark_used(self, fid: str):
        for f in self.facts:
            if f.fact_id == fid:
                f.used = True
                self._global_used[fid] = self._global_used.get(fid, 0) + 1
                break

    def get_unused(self, section_id: int) -> List[AtomicFact]:
        ids = set(self.by_section.get(section_id, []))
        return [
            f for f in self.facts
            if f.fact_id in ids
            and not f.used
            and self._global_used.get(f.fact_id, 0) == 0
        ]

class SentenceDraft(BaseModel):
    draft_id:          str
    text:              str
    committed_passage: str
    fact_id:           str
    section_id:        int

class VerifiedSentence(BaseModel):
    draft_id:          str
    text:              str
    score:             float
    path:              Literal['A', 'B', 'C']
    committed_passage: str
    fact_id:           str
    section_id:        int

class CoherentSection(BaseModel):
    section_id: int
    title:      str
    text:       str
    sentences:  List[VerifiedSentence]
    char_count: int

# ─────────────────────────────────────────────────────────────
# INFRASTRUCTURE (from your notebook §2)
# ─────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_ok = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            if wait > 0:
                time.sleep(wait)
            self._next_ok = time.monotonic() + self._interval

_limiter = _RateLimiter(RATE_LIMIT_RPM)
_groq = Groq(api_key=os.getenv('GROQ_API_KEY', ''))
_embedder = None
_nli = None

def init_models():
    """Initialize embedder and NLI (lazy load)."""
    global _embedder, _nli
    print('Loading embedder and NLI models...')
    _embedder = SentenceTransformer(EMBED_MODEL)
    _nli = CrossEncoder(NLI_MODEL)
    print('✓ Models loaded')

def llm_call(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=GENERATION_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            is_rate = any(x in err for x in ('429','rate_limit','Rate limit'))
            is_transient = any(x in err for x in ('500','503','timeout','unavailable'))
            if (is_rate or is_transient) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                print(f'  ⚠ Retry {attempt+1} in {wait:.1f}s')
                time.sleep(wait)
            else:
                raise

def llm_call_extract(system: str, user: str, max_tokens: int = 1024) -> Any:
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
            )
            raw = resp.choices[0].message.content.strip()
            raw = _repair_json(raw)
            for fence in ('```json', '```'):
                if raw.startswith(fence): raw = raw[len(fence):]
                if raw.endswith('```'): raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as e:
            err = str(e)
            if any(x in err for x in ('429','rate_limit','RESOURCE_EXHAUSTED')) and attempt < MAX_RETRIES-1:
                wait = (2**attempt) + np.random.uniform(0,1)
                time.sleep(wait)
            else:
                raise

def llm_call_small(system: str, user: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{'role': 'system', 'content': system},{'role': 'user', 'content': user}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if any(x in err for x in ('429', 'rate_limit', 'RESOURCE_EXHAUSTED')) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise

def _repair_json(s: str) -> str:
    s = s.strip()
    for fence in ('```json', '```'):
        if s.startswith(fence): s = s[len(fence):]
        if s.endswith('```'): s = s[:-3]
    s = s.strip()
    sd, sl = s.find('{'), s.find('[')
    if sd == -1 and sl == -1: return s
    if sd != -1 and (sl == -1 or sd < sl):
        start, open_c, close_c = sd, '{', '}'
    else:
        start, open_c, close_c = sl, '[', ']'
    s = s[start:]
    depth, in_str, escape, end = 0, False, False, -1
    for i, c in enumerate(s):
        if escape: escape = False; continue
        if c == '\\' and in_str: escape = True; continue
        if c == '"': in_str = not in_str; continue
        if in_str: continue
        if c == open_c: depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0: end = i; break
    if end != -1: return s[:end+1]
    else: return s + close_c * depth

def llm_json(system: str, user: str, max_tokens: int = 2048) -> Any:
    raw = llm_call(system + '\n\nReturn ONLY valid JSON. No markdown fences, no explanation.', user, max_tokens, temperature=0.1)
    cleaned = raw.strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError: pass
    fence = re.sub(r'^```json\n?|^```\n?', '', cleaned, flags=re.IGNORECASE)
    fence = re.sub(r'```$', '', fence).strip()
    try: return json.loads(fence)
    except json.JSONDecodeError: pass
    sd, ed = fence.find('{'), fence.rfind('}')
    sl, el = fence.find('['), fence.rfind(']')
    if sd != -1 and (sl == -1 or sd < sl) and ed > sd:
        try: return json.loads(fence[sd:ed+1])
        except json.JSONDecodeError: pass
    elif sl != -1 and el > sl:
        try: return json.loads(fence[sl:el+1])
        except json.JSONDecodeError: pass
    raise json.JSONDecodeError(f'Could not extract valid JSON. Raw: {raw[:300]}', raw, 0)

def embed(texts): return _embedder.encode([f'query: {t}' for t in texts], normalize_embeddings=True)
def embed_passages(texts): return _embedder.encode([f'passage: {t}' for t in texts], normalize_embeddings=True)
def cosine_sim(a, b): return a @ b.T

def _nli_probs(scores):
    import torch, torch.nn.functional as F
    return F.softmax(torch.tensor(scores), dim=-1)

def nli_score(premise, hypothesis): return float(_nli_probs(_nli.predict([(premise, hypothesis)]))[0][1])
def nli_batch(pairs): return [float(p[1]) for p in _nli_probs(_nli.predict(pairs))]
def split_sentences(text): return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
def fact_id(text, source): return hashlib.md5(f'{source}::{text}'.encode()).hexdigest()[:12]

# ─────────────────────────────────────────────────────────────
# CORPUS (from your notebook §3)
# ─────────────────────────────────────────────────────────────

import pickle

class LocalCorpus:
    EMBED_CHUNK = 2_000

    def __init__(self, cache_size: int = 50_000, language: str = "eng", precomputed_path: str = None):
        self.cache_size = cache_size
        self.language = language
        self.CACHE_FILE = f"/tmp/corpus_cache_{language}_{cache_size}.pkl"
        load_path = precomputed_path or self.CACHE_FILE
        if Path(load_path).exists():
            print(f"[Corpus] Loading {cache_size} docs from cache...")
            with open(load_path, 'rb') as f:
                cache = pickle.load(f)
            self.docs = cache['docs']
            self.embs = cache['embs']
            print(f"[Corpus] Loaded {len(self.docs)} docs")
        else:
            print(f"[Corpus] Building cache ({cache_size} docs)...")
            self._build_cache()

    def _is_quality_doc(self, text: str) -> bool:
        t = text.strip()
        if len(t) < 200: return False
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        if not sents: return False
        avg_words = sum(len(s.split()) for s in sents) / len(sents)
        return avg_words >= 8 and len(sents) >= 3

    def _embed_in_chunks(self, texts: list) -> np.ndarray:
        all_embs = []
        total = len(texts)
        for start in range(0, total, self.EMBED_CHUNK):
            chunk = texts[start : start + self.EMBED_CHUNK]
            chunk_embs = embed_passages(chunk)
            all_embs.append(chunk_embs)
            done = min(start + self.EMBED_CHUNK, total)
            pct = done * 100 // total
            print(f"  {done}/{total} docs ({pct}%)")
        return np.vstack(all_embs)

    def _build_cache(self):
        print("[Corpus] Loading RAGTIME1 dataset...")
        ds = load_dataset("trec-ragtime/ragtime1", data_files="eng-docs.jsonl", split="train")
        print(f"[Corpus] Total docs: {len(ds)}")
        
        import random
        sample_indices = random.sample(range(len(ds)), min(self.cache_size, len(ds)))
        print(f"[Corpus] Sampling {len(sample_indices)}...")
        
        self.docs = []
        for idx in sample_indices:
            doc = ds[idx]
            if not self._is_quality_doc(doc['text']): continue
            self.docs.append({'doc_id': doc['id'], 'text': doc['text'][:1000], 'source': doc.get('url', ''), 'language': 'en'})
            if len(self.docs) % 10000 == 0: print(f"  ... {len(self.docs)} sampled")
        
        print(f"[Corpus] Embedding {len(self.docs)} docs...")
        texts = [d['text'] for d in self.docs]
        self.embs = self._embed_in_chunks(texts)
        
        print(f"[Corpus] Saving cache to {self.CACHE_FILE}...")
        with open(self.CACHE_FILE, 'wb') as f:
            pickle.dump({'docs': self.docs, 'embs': self.embs}, f)

    def search(self, qemb: np.ndarray, top_k: int = 20) -> List[Dict]:
        sims = cosine_sim(qemb.reshape(1, -1), self.embs)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        return [{**self.docs[i], 'score': float(sims[i])} for i in idxs]

# ─────────────────────────────────────────────────────────────
# COMPONENTS C1-C8 (copy from your working notebook)
# ─────────────────────────────────────────────────────────────

class C1_IntentParser:
    SYSTEM = (
        'You are a research report architect.\nGiven a report request, return a single JSON object with:\n'
        '  "goal": one sentence stating what the report must establish\n'
        '  "sections": ordered array of 4-6 objects, each with:\n'
        '      "title": section title\n'
        '      "epistemic_contract": single verb-led sentence stating what this section must ESTABLISH\n'
        '      "retrieval_queries": 2-3 concrete search queries for evidence\n'
        'Respond in the same language as the first preferred language. Return ONLY valid JSON STRICTLY.'
    )

    def run(self, topic: TopicJSON):
        print(f'[C1+C2] {topic.topic_id}')
        user = (f'Title: {topic.title}\nBackground: {topic.background}\nInformation need: {topic.problem_statement}')
        parsed = llm_json(self.SYSTEM, user, max_tokens=2048)
        primary_lang = topic.preferred_languages[0] if topic.preferred_languages else 'en'
        intent = ReportIntent(topic_id=topic.topic_id, goal=parsed['goal'], language=primary_lang,
            section_hints=[s['title'] for s in parsed['sections']], char_budget=topic.limit)
        print(f'  goal: {intent.goal}')
        n = len(parsed['sections'])
        char_each = topic.limit // n
        word_each = char_each // 6
        sent_each = max(3, min(20, char_each // 80))
        plans = []
        for i, s in enumerate(parsed['sections']):
            plans.append(SectionPlan(section_id=i, title=s['title'],
                epistemic_contract=s['epistemic_contract'], word_budget=word_each,
                char_budget=char_each, sentence_budget=sent_each,
                retrieval_queries=s.get('retrieval_queries', [])))
            print(f'  §{i} "{s["title"]}"')
        return intent, plans

class C3_HyDERetriever:
    HYDE_SYSTEM = ('You are a knowledgeable research assistant.\n'
        f'Generate exactly {HYDE_N_HYPS} DISTINCT factual paragraphs (3-4 sentences each) that would satisfy the given epistemic contract.\n'
        'Each paragraph must approach the topic from a different angle.\n'
        f'Return a JSON array of exactly {HYDE_N_HYPS} strings — one paragraph per element. No other text.')

    def __init__(self, corpus): self.corpus = corpus
    def _hypotheticals(self, plan: SectionPlan) -> List[str]:
        raw = llm_call_extract(self.HYDE_SYSTEM, f'Contract: {plan.epistemic_contract}\nSection: {plan.title}', max_tokens=600)
        return [str(h) for h in (raw[:HYDE_N_HYPS] if isinstance(raw, list) else [raw])]
    def _rerank(self, docs: List[Dict], plan: SectionPlan) -> List[Dict]:
        cemb = embed([plan.epistemic_contract])[0]
        dembs = embed_passages([d['text'] for d in docs])
        sims = cosine_sim(cemb.reshape(1,-1), dembs)[0]
        for d, s in zip(docs, sims): d['score'] = float(s)
        return sorted(docs, key=lambda d: d['score'], reverse=True)
    def run(self, plans: List[SectionPlan]) -> Dict[int, List[RetrievedDoc]]:
        results = {}
        for plan in plans:
            print(f'[C3] §{plan.section_id} "{plan.title}"')
            hyps = self._hypotheticals(plan)
            seen = {}
            for hyp in hyps:
                for d in self.corpus.search(embed([hyp])[0], TOP_K_RETRIEVAL):
                    seen.setdefault(d['doc_id'], d)
            reranked = self._rerank(list(seen.values()), plan)
            results[plan.section_id] = [RetrievedDoc(doc_id=d['doc_id'], text=d['text'],
                source=d.get('source','?'), language=d.get('language','en'),
                score=d['score'], section_id=plan.section_id) for d in reranked]
            print(f'  {len(seen)} unique docs')
        return results

class C4_FactPoolBuilder:
    EXTRACT_SYS = ('You are a precise fact extractor. Given numbered passages, extract atomic facts.\n'
        'Output MUST be a single JSON object where keys are the passage numbers.\n'
        'Example: {"1": ["Fact A", "Fact B"], "2": ["Fact C"]}\nStrictly avoid any text before or after the JSON block.')
    DEDUP_THR = 0.92
    SAFE_BATCH_SIZE = 2

    def _extract_batch(self, docs):
        passages = '\n\n'.join(f'Passage {i+1}:\n{doc.text[:250]}' for i, doc in enumerate(docs))
        return llm_call_extract(self.EXTRACT_SYS, passages, max_tokens=800)
    def _dedup(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        if len(facts) <= 1: return facts
        embs = embed_passages([f.text for f in facts])
        sim = cosine_similarity(embs)
        assigned = [False] * len(facts)
        clusters = []
        for i in range(len(facts)):
            if assigned[i]: continue
            cluster = [i]
            assigned[i] = True
            for j in range(i+1, len(facts)):
                if not assigned[j] and sim[i][j] >= self.DEDUP_THR:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return [facts[c[0]] for c in clusters]
    def _is_quality_fact(self, text: str) -> bool:
        t = text.strip()
        if len(t) < 30: return False
        words = t.split()
        if len(words) < 6: return False
        if re.match(r'^[\w\s,:\-\/]+$', t) and len(words) <= 5: return False
        has_verb = any(x in t.lower() for x in [' is ', ' are ', ' was ', ' were ', ' has ', ' have ', ' will ', ' would ', ' can ', ' could ', ' said ', ' found ', ' shows ', ' reveals ', ' warns ', ' reports ', ' claims '])
        return has_verb or len(words) >= 10
    def run(self, topic: TopicJSON, plans: List[SectionPlan], retrieved: Dict[int, List[RetrievedDoc]]) -> FactPool:
        pool = FactPool(topic_id=topic.topic_id)
        for plan in plans:
            docs = retrieved.get(plan.section_id, [])
            print(f'[C4] §{plan.section_id} processing {len(docs)} docs')
            raw_facts = []
            for i in range(0, len(docs), self.SAFE_BATCH_SIZE):
                batch = docs[i: i + self.SAFE_BATCH_SIZE]
                try:
                    result = self._extract_batch(batch)
                    if not isinstance(result, dict): continue
                    for k, v in result.items():
                        if not isinstance(v, list): continue
                        idx = int(k) - 1
                        if not (0 <= idx < len(batch)): continue
                        doc = batch[idx]
                        for fact_text in v:
                            if not isinstance(fact_text, str): continue
                            fid = hashlib.md5(f'{doc.doc_id}:{fact_text}'.encode()).hexdigest()[:12]
                            raw_facts.append(AtomicFact(fact_id=fid, text=fact_text, source_doc_id=doc.doc_id, section_id=plan.section_id))
                except Exception as e:
                    print(f'  ⚠ Batch failed: {e}')
            raw_facts = [f for f in raw_facts if self._is_quality_fact(f.text)]
            deduped = self._dedup(raw_facts)
            pool.by_section[plan.section_id] = [f.fact_id for f in deduped]
            pool.facts.extend(deduped)
        pool.seal()
        print(f'[C4] Pool sealed — {len(pool.facts)} facts')
        return pool

class C5_IsolatedGenerator:
    def run(self, pool: FactPool, plans: List[SectionPlan]) -> List[SentenceDraft]:
        drafts, n = [], 0
        for plan in plans:
            unused = sorted(pool.get_unused(plan.section_id), key=lambda f: f.corroboration, reverse=True)
            selected = unused[:plan.sentence_budget]
            if not selected: continue
            print(f'[C5] §{plan.section_id} — {len(selected)} facts')
            for fact in selected:
                pool.mark_used(fact.fact_id)
                drafts.append(SentenceDraft(draft_id=f'd{n:04d}', text=fact.text,
                    committed_passage=fact.text, fact_id=fact.fact_id, section_id=plan.section_id))
                n += 1
        return drafts

class C6_NLIVerifier:
    def _verbatim(self, passage: str) -> str:
        sents = split_sentences(passage)
        return sents[0] if sents else passage.strip()
    def _verify_one(self, draft: SentenceDraft) -> VerifiedSentence:
        score = nli_score(draft.committed_passage, draft.text)
        if score >= NLI_PATH_A:
            return VerifiedSentence(draft_id=draft.draft_id, text=draft.text, score=score, path='A',
                committed_passage=draft.committed_passage, fact_id=draft.fact_id, section_id=draft.section_id)
        else:
            return VerifiedSentence(draft_id=draft.draft_id, text=self._verbatim(draft.committed_passage), 
                score=0.55, path='C', committed_passage=draft.committed_passage, 
                fact_id=draft.fact_id, section_id=draft.section_id)
    def run(self, drafts: List[SentenceDraft]) -> List[VerifiedSentence]:
        print(f'[C6] Verifying {len(drafts)} sentences')
        return [self._verify_one(d) for d in drafts]

class C7_CoherencePass:
    SYSTEM = ('You are a copy editor. Improve flow between sentences by adjusting punctuation and adding minimal transitions.\n'
        'STRICT: Keep ALL sentences. Do NOT merge, split, or drop. Do NOT rephrase. Return ONLY edited paragraph.')
    def run(self, verified: List[VerifiedSentence], plans: List[SectionPlan]) -> List[CoherentSection]:
        by_sec = {}
        for vs in verified: by_sec.setdefault(vs.section_id, []).append(vs)
        sections = []
        for plan in plans:
            sents = by_sec.get(plan.section_id, [])
            if not sents: continue
            print(f'[C7] §{plan.section_id} "{plan.title}"')
            prompt = '\n'.join(f'Sentence {i+1}: {vs.text}' for i, vs in enumerate(sents))
            raw = llm_call_small(self.SYSTEM, prompt, max_tokens=max(len(sents) * 80, 200))
            sections.append(CoherentSection(section_id=plan.section_id, title=plan.title,
                text=raw, sentences=sents, char_count=len(raw)))
        return sections

class C8_OutputFormatter:
    def run(self, topic: TopicJSON, sections: List[CoherentSection], pool: FactPool) -> dict:
        f2doc = {f.fact_id: f.source_doc_id for f in pool.facts}
        responses = []
        references_set = set()
        for sec in sections:
            for vs in sec.sentences:
                doc_id = f2doc.get(vs.fact_id, 'unknown')
                if doc_id != 'unknown': references_set.add(doc_id)
                responses.append({"text": vs.text, "citations": {doc_id: round(vs.score, 4)},
                    "_meta": {"nli_path": vs.path, "fact_id": vs.fact_id, "section": sec.title, "disputed": False}})
        return {"metadata": {"topic_id": topic.topic_id, "run_id": "EFSG_ACL2026"},
                "responses": responses, "references": list(references_set)}

class EFSGPipeline:
    def __init__(self, corpus):
        self.c1, self.c3 = C1_IntentParser(), C3_HyDERetriever(corpus)
        self.c4, self.c5 = C4_FactPoolBuilder(), C5_IsolatedGenerator()
        self.c6, self.c7, self.c8 = C6_NLIVerifier(), C7_CoherencePass(), C8_OutputFormatter()
    def run(self, topic: TopicJSON):
        t0 = time.time()
        print(f'{"="*60}\nEFSG · {topic.topic_id}\n{"="*60}')
        intent, plans = self.c1.run(topic)
        retrieved = self.c3.run(plans)
        pool = self.c4.run(topic, plans, retrieved)
        drafts = self.c5.run(pool, plans)
        verified = self.c6.run(drafts)
        sections = self.c7.run(verified, plans)
        report = self.c8.run(topic, sections, pool)
        print(f'✓ Done in {time.time()-t0:.1f}s\n')
        return report

# ─────────────────────────────────────────────────────────────
# TIRA ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="efsg-submission")
    parser.add_argument("-i", "--input", required=True, help="Input dir with report-requests.jsonl")
    parser.add_argument("-o", "--output", required=True, help="Output dir for run.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / 'report-requests.jsonl'
    output_file = output_dir / 'run.jsonl'

    if not input_file.exists():
        print(f' {input_file} not found')
        sys.exit(1)

    print('[TIRA] Initializing...')
    init_models()
    corpus = LocalCorpus(cache_size=50_000)
    pipeline = EFSGPipeline(corpus)

    print(f'[TIRA] Processing {input_file}')
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in, 1):
            if not line.strip(): continue
            try:
                topic = TopicJSON(**json.loads(line))
                report = pipeline.run(topic)
                f_out.write(json.dumps(report, ensure_ascii=False) + '\n')
                f_out.flush()
            except Exception as e:
                print(f' Topic {i}: {e}')

    print(f'✓ Output: {output_file}')

if __name__ == '__main__':
    main()