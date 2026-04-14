"""
EFSG Pipeline Components
Extracted from: efsg_pipeline_v3_1_efsg006(duplicates)_c7fix.ipynb
"""

import os
import json
import re
import time
import hashlib
import threading
from typing import List, Optional, Dict, Any, Literal

import numpy as np
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
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
# SCHEMAS
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
# INFRASTRUCTURE
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


def initialize_models():
    """Initialize global LLM, embedder, NLI models."""
    global _groq, _embedder, _nli, _limiter
    
    print('Initializing EFSG infrastructure...')
    _limiter = _RateLimiter(RATE_LIMIT_RPM)
    _groq = Groq(api_key=os.getenv('GROQ_API_KEY', ''))
    print(f'  ✓ Groq LLM: {GENERATION_MODEL}')
    
    _embedder = SentenceTransformer(EMBED_MODEL)
    print(f'  ✓ Embedder: {EMBED_MODEL}')
    
    _nli = CrossEncoder(NLI_MODEL)
    print(f'  ✓ NLI: {NLI_MODEL}')


_groq = None
_embedder = None
_nli = None
_limiter = None


def llm_call(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
    """Call Groq LLM with rate limiting and retries."""
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=GENERATION_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            is_rate = any(x in err for x in ('429', 'rate_limit', 'Rate limit'))
            is_transient = any(x in err for x in ('500', '503', 'timeout', 'unavailable'))
            if (is_rate or is_transient) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                print(f'  ⚠ Retry {attempt+1} in {wait:.1f}s')
                time.sleep(wait)
            else:
                raise
    raise RuntimeError('Max retries exceeded')


def llm_call_extract(system: str, user: str, max_tokens: int = 1024) -> Any:
    """Call extraction model (8B)."""
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}
                ]
            )
            raw = resp.choices[0].message.content.strip()
            raw = _repair_json(raw)
            for fence in ('```json', '```'):
                if raw.startswith(fence): 
                    raw = raw[len(fence):]
                if raw.endswith('```'): 
                    raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as e:
            err = str(e)
            if any(x in err for x in ('429', 'rate_limit', 'RESOURCE_EXHAUSTED')) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(wait)
            else:
                raise


def llm_call_small(system: str, user: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call small model (8B) for coherence."""
    for attempt in range(MAX_RETRIES):
        _limiter.acquire()
        try:
            resp = _groq.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}
                ]
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
    """Repair truncated/malformed JSON."""
    s = s.strip()
    for fence in ('```json', '```'):
        if s.startswith(fence): 
            s = s[len(fence):]
        if s.endswith('```'): 
            s = s[:-3]
    s = s.strip()

    sd, sl = s.find('{'), s.find('[')
    if sd == -1 and sl == -1:
        return s

    if sd != -1 and (sl == -1 or sd < sl):
        start, open_c, close_c = sd, '{', '}'
    else:
        start, open_c, close_c = sl, '[', ']'

    s = s[start:]
    depth, in_str, escape, end = 0, False, False, -1
    for i, c in enumerate(s):
        if escape:            
            escape = False
            continue
        if c == '\\' and in_str: 
            escape = True
            continue
        if c == '"':          
            in_str = not in_str
            continue
        if in_str:            
            continue
        if c == open_c:       
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:    
                end = i
                break

    if end != -1:
        return s[:end+1]
    else:
        return s + close_c * depth


def llm_json(system: str, user: str, max_tokens: int = 2048) -> Any:
    """Call LLM and parse JSON."""
    raw = llm_call(
        system + '\n\nReturn ONLY valid JSON. No markdown fences, no explanation.',
        user, max_tokens, temperature=0.1)
    cleaned = raw.strip()
    
    try: 
        return json.loads(cleaned)
    except json.JSONDecodeError: 
        pass
    
    fence = re.sub(r'^```json\n?|^```\n?', '', cleaned, flags=re.IGNORECASE)
    fence = re.sub(r'```$', '', fence).strip()
    try: 
        return json.loads(fence)
    except json.JSONDecodeError: 
        pass
    
    sd, ed = fence.find('{'), fence.rfind('}')
    sl, el = fence.find('['), fence.rfind(']')
    if sd != -1 and (sl == -1 or sd < sl) and ed > sd:
        try: 
            return json.loads(fence[sd:ed+1])
        except json.JSONDecodeError: 
            pass
    elif sl != -1 and el > sl:
        try: 
            return json.loads(fence[sl:el+1])
        except json.JSONDecodeError: 
            pass
    
    raise json.JSONDecodeError(f'Could not extract valid JSON. Raw: {raw[:300]}', raw, 0)


def embed(texts):          
    return _embedder.encode([f'query: {t}'   for t in texts], normalize_embeddings=True)


def embed_passages(texts): 
    return _embedder.encode([f'passage: {t}' for t in texts], normalize_embeddings=True)


def cosine_sim(a, b):      
    return a @ b.T


def _nli_probs(scores):
    import torch
    import torch.nn.functional as F
    return F.softmax(torch.tensor(scores), dim=-1)


def nli_score(premise, hypothesis): 
    return float(_nli_probs(_nli.predict([(premise, hypothesis)]))[0][1])


def nli_batch(pairs):               
    return [float(p[1]) for p in _nli_probs(_nli.predict(pairs))]


def split_sentences(text): 
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


def fact_id(text, source): 
    return hashlib.md5(f'{source}::{text}'.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# COMPONENTS (C1-C8)
# ─────────────────────────────────────────────────────────────

class C1_IntentParser:
    SYSTEM = (
        'You are a research report architect.\n'
        'Given a report request, return a single JSON object with:\n'
        '  "goal": one sentence stating what the report must establish\n'
        '  "sections": ordered array of 4-6 objects, each with:\n'
        '      "title": section title\n'
        '      "epistemic_contract": single verb-led sentence stating what this section must ESTABLISH\n'
        '      "retrieval_queries": 2-3 concrete search queries for evidence\n'
        'Respond in the same language as the first preferred language. Return ONLY valid JSON STRICTLY.'
    )

    def run(self, topic: TopicJSON):
        print(f'[C1+C2] {topic.topic_id}')
        user = (
            f'Title: {topic.title}\n'
            f'Background: {topic.background}\n'
            f'Information need: {topic.problem_statement}'
        )
        parsed = llm_json(self.SYSTEM, user, max_tokens=2048)
        primary_lang = topic.preferred_languages[0] if topic.preferred_languages else 'en'
        
        intent = ReportIntent(
            topic_id=topic.topic_id,
            goal=parsed['goal'],
            language=primary_lang,
            section_hints=[s['title'] for s in parsed['sections']],
            char_budget=topic.limit,
        )
        print(f'  goal: {intent.goal}')
        print(f'  char_budget: {intent.char_budget}')

        n         = len(parsed['sections'])
        char_each = topic.limit // n
        word_each = char_each // 6
        sent_each = max(3, min(20, char_each // 80))

        plans = []
        for i, s in enumerate(parsed['sections']):
            plans.append(SectionPlan(
                section_id=i, title=s['title'],
                epistemic_contract=s['epistemic_contract'],
                word_budget=word_each, char_budget=char_each, sentence_budget=sent_each,
                retrieval_queries=s.get('retrieval_queries', []),
            ))
            print(f'  §{i} "{s["title"]}"')

        print(f'  → 1 API call | {n} sections | {char_each} chars each')
        return intent, plans


class C3_HyDERetriever:
    HYDE_SYSTEM = (
        'You are a knowledgeable research assistant.\n'
        f'Generate exactly {HYDE_N_HYPS} DISTINCT factual paragraphs (3-4 sentences each) '
        'that would satisfy the given epistemic contract.\n'
        'Each paragraph must approach the topic from a different angle.\n'
        'Return a JSON array of exactly {n} strings — one paragraph per element. No other text.'
        .replace('{n}', str(HYDE_N_HYPS))
    )

    def __init__(self, corpus): 
        self.corpus = corpus

    def _hypotheticals(self, plan: SectionPlan) -> List[str]:
        raw = llm_call_extract(
            self.HYDE_SYSTEM,
            f'Contract: {plan.epistemic_contract}\nSection: {plan.title}',
            max_tokens=600,
        )
        if isinstance(raw, list) and len(raw) >= 1:
            return [str(h) for h in raw[:HYDE_N_HYPS]]
        return [str(raw)]

    def _rerank(self, docs: List[Dict], plan: SectionPlan) -> List[Dict]:
        cemb  = embed([plan.epistemic_contract])[0]
        dembs = embed_passages([d['text'] for d in docs])
        sims  = cosine_sim(cemb.reshape(1,-1), dembs)[0]
        for d, s in zip(docs, sims): 
            d['score'] = float(s)
        return sorted(docs, key=lambda d: d['score'], reverse=True)

    def run(self, plans: List[SectionPlan]) -> Dict[int, List[RetrievedDoc]]:
        results = {}
        for plan in plans:
            print(f'\n[C3] §{plan.section_id} "{plan.title}"')
            hyps = self._hypotheticals(plan)
            print(f'  {len(hyps)} hypotheticals (1 API call)')
            seen = {}
            for hyp in hyps:
                for d in self.corpus.search(embed([hyp])[0], TOP_K_RETRIEVAL):
                    seen.setdefault(d['doc_id'], d)
            reranked = self._rerank(list(seen.values()), plan)
            results[plan.section_id] = [
                RetrievedDoc(doc_id=d['doc_id'], text=d['text'], source=d.get('source','?'),
                             language=d.get('language','en'), score=d['score'], section_id=plan.section_id)
                for d in reranked
            ]
            print(f'  {len(seen)} unique docs | top: "{results[plan.section_id][0].source}" '
                  f'({results[plan.section_id][0].score:.3f})')
        return results


class C4_FactPoolBuilder:
    EXTRACT_SYS = (
        'You are a precise fact extractor. Given numbered passages, extract atomic facts.\n'
        'Output MUST be a single JSON object where keys are the passage numbers.\n'
        'Example: {"1": ["Fact A", "Fact B"], "2": ["Fact C"]}\n'
        'Strictly avoid any text before or after the JSON block.'
    )
    DEDUP_THR      = 0.92
    GLOBAL_THR     = 0.88
    SAFE_BATCH_SIZE = 2

    def _extract_batch(self, docs):
        passages = '\n\n'.join(
            f'Passage {i+1}:\n{doc.text[:250]}'
            for i, doc in enumerate(docs)
        )
        return llm_call_extract(self.EXTRACT_SYS, passages, max_tokens=800)

    def _dedup(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        if len(facts) <= 1: 
            return facts
        embs = embed_passages([f.text for f in facts])
        sim  = cosine_similarity(embs)
        assigned = [False] * len(facts)
        clusters = []
        for i in range(len(facts)):
            if assigned[i]: 
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i+1, len(facts)):
                if not assigned[j] and sim[i][j] >= self.DEDUP_THR:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return [facts[c[0]] for c in clusters]

    def _global_dedup(self, pool: FactPool) -> int:
        if len(pool.facts) <= 1:
            return 0

        texts = [f.text for f in pool.facts]
        embs  = embed_passages(texts)
        sim   = cosine_similarity(embs)

        n      = len(pool.facts)
        kept   = [False] * n
        marked = 0

        for i in range(n):
            if pool.facts[i].used:
                continue
            is_dup = False
            for j in range(i):
                if kept[j] and sim[i][j] >= self.GLOBAL_THR:
                    is_dup = True
                    break
            if is_dup:
                pool.facts[i].used = True
                marked += 1
            else:
                kept[i] = True

        return marked

    def _is_quality_fact(self, text: str) -> bool:
        t = text.strip()
        if len(t) < 30: 
            return False
        words = t.split()
        if len(words) < 6: 
            return False
        if re.match(r'^[\w\s,:\-\/]+$', t) and len(words) <= 5: 
            return False
        has_verb = any(x in t.lower() for x in [
            ' is ', ' are ', ' was ', ' were ', ' has ', ' have ',
            ' will ', ' would ', ' can ', ' could ', ' said ', ' found ',
            ' shows ', ' reveals ', ' warns ', ' reports ', ' claims '
        ])
        if not has_verb and len(words) < 10: 
            return False
        return True

    def run(self, topic: TopicJSON, plans: List[SectionPlan],
            retrieved: Dict[int, List[RetrievedDoc]]) -> FactPool:
        pool = FactPool(topic_id=topic.topic_id)

        for plan in plans:
            docs = retrieved.get(plan.section_id, [])
            print(f'\n[C4] §{plan.section_id} processing {len(docs)} docs')
            raw_facts: List[AtomicFact] = []
            target = int(plan.sentence_budget * FACT_POOL_RATIO * 6)

            for i in range(0, len(docs), self.SAFE_BATCH_SIZE):
                if len(raw_facts) >= target:
                    break
                batch = docs[i: i + self.SAFE_BATCH_SIZE]
                try:
                    result = self._extract_batch(batch)
                except Exception as e:
                    print(f'  ⚠ Batch {i//self.SAFE_BATCH_SIZE+1} failed: {e}')
                    continue

                if not isinstance(result, dict): 
                    continue

                for k, v in result.items():
                    if not isinstance(v, list): 
                        continue
                    idx = int(k) - 1
                    if not (0 <= idx < len(batch)): 
                        continue
                    doc = batch[idx]
                    for fact_text in v:
                        if not isinstance(fact_text, str): 
                            continue
                        fid = hashlib.md5(
                            f'{doc.doc_id}:{fact_text}'.encode()
                        ).hexdigest()[:12]
                        raw_facts.append(AtomicFact(
                            fact_id=fid, text=fact_text,
                            source_doc_id=doc.doc_id, section_id=plan.section_id
                        ))

            raw_facts = [f for f in raw_facts if self._is_quality_fact(f.text)]
            deduped   = self._dedup(raw_facts)

            pool.by_section[plan.section_id] = [f.fact_id for f in deduped]
            pool.facts.extend(deduped)

        n_marked = self._global_dedup(pool)
        if n_marked:
            print(f'\n[C4] Global dedup: pre-marked {n_marked} cross-section duplicates')

        pool.seal()
        available = sum(1 for f in pool.facts if not f.used)
        print(f'\n[C4] Pool sealed — {len(pool.facts)} total | {available} available after dedup')
        return pool


class C5_IsolatedGenerator:
    SYSTEM_TIGHT = (
        'You are a sentence writer. Write one sentence using ONLY words and numbers '
        'that appear verbatim in the passage. Return ONLY the sentence.'
    )

    def _gen(self, passage: str, system: str = None, temperature: float = 0.1) -> str:
        return llm_call(
            system or self.SYSTEM_TIGHT,
            f'Passage:\n{passage}',
            max_tokens=150, temperature=temperature
        )

    def run(self, pool: FactPool, plans: List[SectionPlan]) -> List[SentenceDraft]:
        assert pool.sealed, 'Pool must be sealed before C5 (Phase Boundary violated)'
        drafts, n = [], 0
        for plan in plans:
            unused   = sorted(pool.get_unused(plan.section_id),
                               key=lambda f: f.corroboration, reverse=True)
            selected = unused[:plan.sentence_budget]
            if not selected:
                continue
            print(f'\n[C5] §{plan.section_id} — {len(selected)} facts → sentences (0 API calls)')
            for fact in selected:
                pool.mark_used(fact.fact_id)
                drafts.append(SentenceDraft(
                    draft_id=f'd{n:04d}',
                    text=fact.text,
                    committed_passage=fact.text,
                    fact_id=fact.fact_id,
                    section_id=plan.section_id,
                ))
                n += 1
            print(f'  → {len(selected)} drafts')
        print(f'\n[C5] Total: {n} drafts | API calls: 0')
        return drafts


class C6_NLIVerifier:
    def _verbatim(self, passage: str) -> str:
        sents = split_sentences(passage)
        return sents[0] if sents else passage.strip()

    def _verify_one(self, draft: SentenceDraft, c5: C5_IsolatedGenerator) -> VerifiedSentence:
        p, h = draft.committed_passage, draft.text
        score = nli_score(p, h)
        if score >= NLI_PATH_A:
            return VerifiedSentence(draft_id=draft.draft_id, text=h, score=score, path='A',
                committed_passage=p, fact_id=draft.fact_id, section_id=draft.section_id)
        elif score >= NLI_PATH_B:
            retry = c5._gen(p, system=c5.SYSTEM_TIGHT, temperature=0.1)
            rs    = nli_score(p, retry)
            text  = retry if rs > score else h
            return VerifiedSentence(draft_id=draft.draft_id, text=text, score=max(score,rs), path='B',
                committed_passage=p, fact_id=draft.fact_id, section_id=draft.section_id)
        else:
            return VerifiedSentence(draft_id=draft.draft_id, text=self._verbatim(p), score=0.55, path='C',
                committed_passage=p, fact_id=draft.fact_id, section_id=draft.section_id)

    def run(self, drafts: List[SentenceDraft], c5: C5_IsolatedGenerator) -> List[VerifiedSentence]:
        print(f'\n[C6] Verifying {len(drafts)} sentences …')
        t0 = time.time()
        verified = [self._verify_one(d, c5) for d in drafts]
        paths = {'A':0,'B':0,'C':0}
        for v in verified: 
            paths[v.path] += 1
        print(f'[C6] Done in {time.time()-t0:.1f}s | A={paths["A"]} B={paths["B"]} C={paths["C"]}')
        return verified


class C7_CoherencePass:
    SYSTEM = (
        'You are a copy editor for a factual research report.\n'
        'Improve the flow between the given sentences by:\n'
        '  - Adjusting punctuation at sentence boundaries\n'
        '  - Adding a minimal transition word (e.g. "Additionally", "However") '
        'at the START of a sentence only — these must assert nothing new\n'
        '  - Fixing obvious grammatical errors\n'
        'STRICT RULES:\n'
        '  - Keep ALL sentences. Do NOT merge, split, or drop any.\n'
        '  - Do NOT rephrase, paraphrase, or change factual content.\n'
        '  - Do NOT add facts, context, or background knowledge.\n'
        'Return ONLY the edited paragraph. No preamble, no commentary.'
    )

    def _sweep(self, text: str, originals: List[VerifiedSentence]) -> str:
        sents  = split_sentences(text)
        n_orig = len(originals)

        if not (n_orig - 2 <= len(sents) <= n_orig + 1):
            print(f'  ⚠ Count {len(sents)} vs {n_orig} outside tolerance — reverting section')
            return ' '.join(v.text for v in originals)

        final, reverted = [], 0
        for i, cs in enumerate(sents):
            orig  = originals[min(i, n_orig - 1)]
            score = nli_score(orig.committed_passage, cs)
            if score >= NLI_COHERENCE_FLOOR:
                final.append(cs)
            else:
                reverted += 1
                final.append(orig.text)

        if reverted:
            print(f'  ⚠ Reverted {reverted}/{len(sents)} sentences (score < {NLI_COHERENCE_FLOOR})')
        return ' '.join(final)

    def run(self, verified: List[VerifiedSentence],
            plans: List[SectionPlan]) -> List[CoherentSection]:
        by_sec: Dict[int, List[VerifiedSentence]] = {}
        for vs in verified:
            by_sec.setdefault(vs.section_id, []).append(vs)

        sections = []
        for plan in plans:
            sents = by_sec.get(plan.section_id, [])
            if not sents: 
                continue
            print(f'\n[C7] §{plan.section_id} "{plan.title}" ({len(sents)} sentences)')

            prompt = '\n'.join(
                f'Sentence {i+1}: {vs.text}'
                for i, vs in enumerate(sents)
            )
            raw   = llm_call_small(self.SYSTEM, prompt,
                                   max_tokens=max(len(sents) * 80, 200))
            final = self._sweep(raw, sents)

            sections.append(CoherentSection(
                section_id=plan.section_id, title=plan.title,
                text=final, sentences=sents, char_count=len(final)
            ))
            print(f'  → {len(final)} chars')

        return sections


class C8_OutputFormatter:
    def _fact_to_doc(self, pool: FactPool) -> Dict[str, str]:
        return {f.fact_id: f.source_doc_id for f in pool.facts}

    def run(self, topic: TopicJSON, sections: List[CoherentSection],
            pool: FactPool, run_id: str = "EFSG_Team_01") -> dict:

        char_limit = topic.limit
        f2doc = self._fact_to_doc(pool)

        total, kept = 0, []
        for sec in sections:
            if total + sec.char_count <= char_limit:
                kept.append(sec)
                total += sec.char_count
            else:
                rem = char_limit - total
                t2  = sec.text[:rem].rsplit(' ', 1)[0]
                s2  = sec.model_copy()
                s2.text = t2
                s2.char_count = len(t2)
                kept.append(s2)
                total += len(t2)
                print(f'[C8] §{sec.section_id} trimmed to fit {char_limit} char limit')
                break

        responses = []
        references_set = set()

        plan_title_map = {sec.section_id: sec.title for sec in kept}
        for sec in kept:
            for vs in sec.sentences:
                doc_id = f2doc.get(vs.fact_id, 'unknown')
                if doc_id != 'unknown':
                    references_set.add(doc_id)

                responses.append({
                    "text": vs.text,
                    "citations": {doc_id: round(vs.score, 4)},
                    "_meta": {
                        "nli_path": vs.path,
                        "fact_id":  vs.fact_id,
                        "section":  plan_title_map.get(vs.section_id, ""),
                        "disputed": False
                    }
                })

        report_json = {
            "metadata": {
                "topic_id": topic.topic_id,
                "run_id": run_id
            },
            "responses": responses,
            "references": list(references_set)
        }

        print(f'\n[C8] ━━━ OUTPUT COMPLETE ━━━')
        print(f'  Sentences: {len(responses)} | Refs: {len(references_set)} | Chars: {total}/{char_limit}')

        return report_json


class EFSGPipeline:
    def __init__(self, corpus):
        self.c1  = C1_IntentParser()
        self.c3  = C3_HyDERetriever(corpus)
        self.c4  = C4_FactPoolBuilder()
        self.c5  = C5_IsolatedGenerator()
        self.c6  = C6_NLIVerifier()
        self.c7  = C7_CoherencePass()
        self.c8  = C8_OutputFormatter()

    def run(self, topic: TopicJSON, run_id: str = "EFSG_Team_01"):
        t0 = time.time()
        print('='*60)
        print(f'EFSG · topic={topic.topic_id} · limit={topic.limit}')
        print('='*60)

        intent, plans = self.c1.run(topic)
        retrieved = self.c3.run(plans)
        pool      = self.c4.run(topic, plans, retrieved)

        print('\n' + '─'*60)
        print('PHASE BOUNDARY CROSSED — generation begins')
        print('─'*60)

        drafts   = self.c5.run(pool, plans)
        verified = self.c6.run(drafts, self.c5)
        sections = self.c7.run(verified, plans)
        report_json = self.c8.run(topic, sections, pool, run_id=run_id)

        print(f'\n✓ Pipeline complete in {time.time()-t0:.1f}s')
        return report_json, pool, plans