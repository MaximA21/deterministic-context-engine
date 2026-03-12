"""Context engine with append-only ChunkLog, compaction, and CerebrasSession wrapper.

Manages what stays in LLM context via:
- Append-only logging with SHA-256 content-addressing
- Soft/hard threshold compaction
- DecisionRecords for auditability
- SQLite WAL for persistence
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~3.2 chars per token.

    Cerebras's actual tokenizer counts ~17% more tokens than len/4.
    Using len/3.2 aligns with measured Cerebras token counts.
    """
    return max(1, int(len(text) / 3.2))


# --- AutoPriority ---

_RE_FILENAMES = re.compile(r'\b[\w\-]+\.(?:py|js|ts|rs|go|java|c|cpp|h|rb|sh|yaml|yml|json|sql|toml|cfg|md|txt|html|css)\b')
_RE_FUNC_CLASS = re.compile(r'\b(?:def|class|function|fn)\s+(\w+)')
_RE_ERROR_INDICATORS = re.compile(r'\b(?:Error|Exception|bug|fix|fail|issue|traceback|panic|IMPORTANT|CRITICAL|UPDATE)\b', re.IGNORECASE)
_RE_QUOTED = re.compile(r'(?:`([^`]+)`|"([^"]+)"|\'([^\']+)\')')
_RE_IP_ADDR = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_RE_DATE_PATTERNS = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b', re.IGNORECASE)


def extract_keywords(message: str) -> set[str]:
    """Extract searchable keywords from a message using regex patterns.

    Extracts: filenames, function/class names, error indicators,
    quoted/backtick contents, IP addresses, date patterns.
    """
    keywords: set[str] = set()

    # Filenames
    for m in _RE_FILENAMES.finditer(message):
        keywords.add(m.group(0).lower())

    # Function/class names
    for m in _RE_FUNC_CLASS.finditer(message):
        keywords.add(m.group(1).lower())

    # Error indicators
    for m in _RE_ERROR_INDICATORS.finditer(message):
        keywords.add(m.group(0).lower())

    # Quoted and backtick contents
    for m in _RE_QUOTED.finditer(message):
        content = m.group(1) or m.group(2) or m.group(3)
        if content and len(content) > 2:
            keywords.add(content.lower())

    # IP addresses
    for m in _RE_IP_ADDR.finditer(message):
        keywords.add(m.group(0))

    # Date patterns
    for m in _RE_DATE_PATTERNS.finditer(message):
        keywords.add(m.group(0).lower())

    # Also extract individual significant words from the message (nouns, proper names)
    # Look for capitalized words that aren't at sentence starts
    words = message.split()
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if clean and len(clean) > 2:
            # Numbers that look like specific values (e.g., "250", "3am")
            if re.match(r'^\d+[a-z]*$', clean) and len(clean) <= 8:
                keywords.add(clean.lower())

    return keywords


def score_chunk(chunk_text: str, keywords: set[str]) -> float:
    """Score a chunk based on keyword matches.

    Returns priority: 0.5 (no matches) to 2.0 (3+ matches).
    Linear interpolation between.
    """
    if not keywords:
        return 0.5

    chunk_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw in chunk_lower)

    if matches == 0:
        return 0.5
    elif matches >= 3:
        return 2.0
    else:
        # Linear interpolation: 0 matches -> 0.5, 3 matches -> 2.0
        return 0.5 + (matches / 3.0) * 1.5


# --- Entity Extractor (pure regex, no ML) ---

class EntityExtractor:
    """Extracts named entities from structured text using regex patterns.

    Targets: SQL table names, CLI flag values, env var assignments,
    JSON key-value pairs, IP addresses, ports, URLs, file paths,
    version numbers, commit hashes, API key patterns, error codes.

    Pure regex — no ML, under 1ms for typical chunks.
    """

    _RE_SQL_TABLE = re.compile(
        r'\b(?:CREATE\s+TABLE|INSERT\s+INTO|UPDATE|DELETE\s+FROM|FROM|JOIN|ALTER\s+TABLE|DROP\s+TABLE|INTO)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(\w+)',
        re.IGNORECASE,
    )
    _RE_CLI_FLAG_EQ = re.compile(r'--(\w[\w-]*)=([\w./:@_-]+)')
    _RE_ENV_VAR_ASSIGN = re.compile(r'\b([A-Z][A-Z0-9_]{2,})=([\S]+)')
    _RE_ENV_VAR_NAME = re.compile(r'\b([A-Z][A-Z0-9_]{3,})\b')
    _RE_EXPORT_VAR = re.compile(r'\bexport\s+([A-Z][A-Z0-9_]{2,})=([\S]+)')
    _RE_IP_PORT = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d{1,5})?)\b')
    _RE_URL = re.compile(r'https?://[\w./:@?&=%-]+')
    _RE_ENDPOINT = re.compile(r'/api/[\w./-]+')
    _RE_FILE_PATH = re.compile(r'(?:^|[\s(])(/[\w./-]+\.\w+|[\w./-]+\.(?:py|js|ts|rs|go|java|yaml|yml|json|sql|toml|proto|conf|cfg|cpp|h))\b')
    _RE_VERSION = re.compile(r'\b[vV]?\d+\.\d+(?:\.\d+)(?:-[\w.]+)?\b')
    _RE_API_KEY = re.compile(r'\b(?:sk_live_|sk_test_|whsec_|ca_|pk_live_|pk_test_|rk_live_)[\w]+')
    _RE_ERROR_CODE = re.compile(r'\b[A-Z]{2,}[-_]\d{3,}(?:[-_][A-Z]+(?:[-_][A-Z]+)*)?\b')
    _RE_NAMESPACE = re.compile(r'--namespace[=\s]([\w-]+)')
    _RE_TICKET = re.compile(r'\b[A-Z]{2,}-\d{3,}\b')
    _RE_JSON_KEY_NAME = re.compile(r'"(\w[\w_-]{2,})"')
    _RE_SNAKE_IDENT = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b')

    # Common words to exclude from identifier extraction
    _STOP_WORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "that", "this",
        "with", "from", "they", "been", "have", "many", "some", "them",
        "than", "its", "over", "such", "into", "other", "what", "which",
        "their", "will", "each", "about", "would", "make", "like",
        "create", "table", "select", "insert", "update", "delete", "index",
        "check", "default", "not_null", "primary", "unique", "where",
        "null", "true", "false", "string", "integer", "boolean",
        "statement", "required", "value", "field", "exact", "must",
        "our", "your", "any", "set", "get", "new", "old", "key",
        # Structural UPPER_CASE names (appear in many templates)
        "error_code", "severity", "message", "action", "dashboard",
        "important", "critical", "required", "uuid", "varchar", "text",
        "timestamptz", "constraint", "references", "chunks",
    })

    def extract_entities(self, text: str) -> set[str]:
        """Extract named entities from text. Returns lowercased entity strings."""
        entities: set[str] = set()

        # SQL table names
        for m in self._RE_SQL_TABLE.finditer(text):
            name = m.group(1).lower()
            if name not in self._STOP_WORDS and len(name) > 2:
                entities.add(name)

        # CLI flags with = (--flag=value): extract value only, not flag name
        for m in self._RE_CLI_FLAG_EQ.finditer(text):
            val = m.group(2).lower()
            if len(val) > 2:
                entities.add(val)

        # Env var assignments (KEY=VALUE)
        for m in self._RE_ENV_VAR_ASSIGN.finditer(text):
            entities.add(m.group(1).lower())
            entities.add(m.group(2).lower())

        # Bare env var names (UPPER_CASE without =)
        for m in self._RE_ENV_VAR_NAME.finditer(text):
            name = m.group(1)
            if name.lower() not in self._STOP_WORDS:
                entities.add(name.lower())

        # export KEY=VALUE
        for m in self._RE_EXPORT_VAR.finditer(text):
            entities.add(m.group(1).lower())

        # API endpoints (/api/...)
        for m in self._RE_ENDPOINT.finditer(text):
            entities.add(m.group(0).lower())

        # IP:port
        for m in self._RE_IP_PORT.finditer(text):
            entities.add(m.group(1))

        # URLs
        for m in self._RE_URL.finditer(text):
            entities.add(m.group(0).lower())

        # File paths
        for m in self._RE_FILE_PATH.finditer(text):
            entities.add(m.group(1).lower())

        # Version numbers
        for m in self._RE_VERSION.finditer(text):
            entities.add(m.group(0).lower())

        # API keys / secrets
        for m in self._RE_API_KEY.finditer(text):
            entities.add(m.group(0).lower())

        # Error codes (PAY-4012-RETRY-EXHAUSTED)
        for m in self._RE_ERROR_CODE.finditer(text):
            entities.add(m.group(0).lower())

        # Namespace
        for m in self._RE_NAMESPACE.finditer(text):
            entities.add(m.group(1).lower())

        # Tickets (INC-7734, SRE-2291, etc.)
        for m in self._RE_TICKET.finditer(text):
            entities.add(m.group(0).lower())

        # JSON key names (anything in quotes before a colon)
        for m in self._RE_JSON_KEY_NAME.finditer(text):
            name = m.group(1).lower()
            if name not in self._STOP_WORDS:
                entities.add(name)

        # Snake_case identifiers (billing_events, idempotency_key, etc.)
        for m in self._RE_SNAKE_IDENT.finditer(text):
            ident = m.group(1).lower()
            if ident not in self._STOP_WORDS and len(ident) > 4:
                entities.add(ident)

        return entities


# --- Structural Fingerprinter ---

class StructuralFingerprinter:
    """Extracts structural tokens from text, capturing SHAPE separately from content.

    Generates a set of synthetic tokens describing text structure:
    - Action/urgency markers (CRITICAL, URGENT, FIX, etc.)
    - Specific value patterns (line numbers, IPs, version numbers, times)
    - Change instructions ("change X from Y to Z")
    - JSON/code nesting depth
    - SQL keyword patterns
    - Text organization (sentence length, paragraph structure, lists)
    - Code patterns (definitions, indentation, comparisons)

    Two chunks with the same structure but different values produce different
    fingerprints because the structural token SET differs at granular level.
    """

    _RE_LINE_REF = re.compile(r'\bline\s+\d+', re.IGNORECASE)
    _RE_ACTION_MARKERS = re.compile(
        r'\b(CRITICAL|URGENT|ALERT|BUG|FIX|ACTION\s+REQUIRED|SECURITY|INCIDENT|IMPORTANT|UPDATE)\b',
        re.IGNORECASE,
    )
    _RE_CHANGE_PATTERN = re.compile(
        r'\b(?:change|replace|update|modify|switch)\s+.*?\b(?:from|to)\b', re.IGNORECASE
    )
    _RE_MUST_PATTERN = re.compile(
        r'\b(?:must|needs?\s+to|required?\s+to|has\s+to)\b', re.IGNORECASE
    )
    _RE_JSON_BRACE = re.compile(r'[{}]')
    _RE_SQL_KEYWORDS = re.compile(
        r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|FROM|WHERE|JOIN|INDEX)\b',
        re.IGNORECASE,
    )
    _RE_CODE_DEF = re.compile(r'\b(?:def|class|function|fn|func|var|let|const)\b')
    _RE_INDENT = re.compile(r'^(?:  {2,}|\t+)', re.MULTILINE)
    _RE_BULLET = re.compile(r'^\s*[-*•]\s', re.MULTILINE)
    _RE_NUMBERED = re.compile(r'^\s*\d+[.)]\s', re.MULTILINE)
    _RE_SPECIFIC_NUM = re.compile(r'\b\d{2,}\b')
    _RE_IP_ADDR = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    _RE_VERSION = re.compile(r'\b[vV]?\d+\.\d+(?:\.\d+)?\b')
    _RE_TIME_PATTERN = re.compile(r'\b\d{1,2}(?:am|pm|:\d{2})\b', re.IGNORECASE)
    _RE_COMPARISON = re.compile(r'[<>=!]=|<=|>=')
    _RE_CODE_BLOCK = re.compile(r'```')
    _RE_EMPLOYEE_ID = re.compile(r'\b[A-Z]{1,3}-\d{3,}\b')
    _RE_MEMORY_SIZE = re.compile(r'\b\d+(?:MB|GB|KB|TB)(?:/\w+)?\b', re.IGNORECASE)

    def extract_structural_tokens(self, text: str) -> frozenset[str]:
        """Extract structural token set describing text shape.

        Returns frozenset of tokens like:
        {"STRUCT_HAS_LINE_REF", "STRUCT_ACTION_CRITICAL", "STRUCT_NUM_DENSE", ...}
        """
        tokens: list[str] = []

        # Line references ("line 42", "line 187")
        line_refs = len(self._RE_LINE_REF.findall(text))
        if line_refs >= 2:
            tokens.append("STRUCT_MULTI_LINE_REF")
        elif line_refs == 1:
            tokens.append("STRUCT_HAS_LINE_REF")

        # Action/urgency markers
        action_matches = self._RE_ACTION_MARKERS.findall(text)
        unique_actions = {m.upper().replace(' ', '_') for m in action_matches}
        for marker in unique_actions:
            tokens.append(f"STRUCT_ACTION_{marker}")
        if len(unique_actions) >= 2:
            tokens.append("STRUCT_MULTI_ACTION")

        # Change instructions ("change X from Y to Z")
        if self._RE_CHANGE_PATTERN.search(text):
            tokens.append("STRUCT_HAS_CHANGE_INSTRUCTION")

        # Imperative/obligation patterns ("must", "needs to")
        must_count = len(self._RE_MUST_PATTERN.findall(text))
        if must_count >= 2:
            tokens.append("STRUCT_STRONG_OBLIGATION")
        elif must_count == 1:
            tokens.append("STRUCT_HAS_OBLIGATION")

        # JSON nesting depth
        braces = self._RE_JSON_BRACE.findall(text)
        depth = 0
        max_depth = 0
        for b in braces:
            depth += 1 if b == '{' else -1
            max_depth = max(max_depth, depth)
        if max_depth >= 3:
            tokens.append("STRUCT_JSON_DEEP")
        elif max_depth >= 1:
            tokens.append("STRUCT_JSON_SHALLOW")

        # SQL keyword pattern
        sql_kws = {m.upper() for m in self._RE_SQL_KEYWORDS.findall(text)}
        if len(sql_kws) >= 3:
            tokens.append("STRUCT_SQL_COMPLEX")
        elif sql_kws:
            tokens.append("STRUCT_SQL_PRESENT")

        # Code definitions
        if self._RE_CODE_DEF.search(text):
            tokens.append("STRUCT_HAS_CODE_DEF")

        # Indentation depth (code-like structure)
        indent_count = len(self._RE_INDENT.findall(text))
        if indent_count > 5:
            tokens.append("STRUCT_DEEP_INDENT")
        elif indent_count > 0:
            tokens.append("STRUCT_HAS_INDENT")

        # List structure
        list_items = len(self._RE_BULLET.findall(text)) + len(self._RE_NUMBERED.findall(text))
        if list_items > 3:
            tokens.append("STRUCT_LIST_HEAVY")
        elif list_items > 0:
            tokens.append("STRUCT_HAS_LIST")

        # Specific number density
        specific_nums = self._RE_SPECIFIC_NUM.findall(text)
        word_count = max(len(text.split()), 1)
        num_density = len(specific_nums) / word_count
        if num_density > 0.04:
            tokens.append("STRUCT_NUM_DENSE")
        elif len(specific_nums) > 2:
            tokens.append("STRUCT_HAS_NUMS")

        # IP addresses
        if self._RE_IP_ADDR.search(text):
            tokens.append("STRUCT_HAS_IP")

        # Version numbers
        if self._RE_VERSION.search(text):
            tokens.append("STRUCT_HAS_VERSION")

        # Time references
        if self._RE_TIME_PATTERN.search(text):
            tokens.append("STRUCT_HAS_TIME")

        # Code blocks (```)
        if self._RE_CODE_BLOCK.search(text):
            tokens.append("STRUCT_HAS_CODE_BLOCK")

        # Comparison operators in text (<=, >=, !=)
        if self._RE_COMPARISON.search(text):
            tokens.append("STRUCT_HAS_COMPARISON")

        # Employee/ticket IDs
        if self._RE_EMPLOYEE_ID.search(text):
            tokens.append("STRUCT_HAS_IDENTIFIER")

        # Memory sizes (50MB/hour, 10GB)
        if self._RE_MEMORY_SIZE.search(text):
            tokens.append("STRUCT_HAS_MEMORY_SIZE")

        # Text density: average sentence length (words per sentence)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sent_len > 18:
            tokens.append("STRUCT_LONG_SENTENCES")
        elif avg_sent_len < 10:
            tokens.append("STRUCT_SHORT_SENTENCES")

        # Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 4:
            tokens.append("STRUCT_MANY_PARAGRAPHS")
        elif len(paragraphs) <= 1:
            tokens.append("STRUCT_SINGLE_BLOCK")

        # Overall text length bucket
        if word_count > 150:
            tokens.append("STRUCT_LONG_TEXT")
        elif word_count < 60:
            tokens.append("STRUCT_SHORT_TEXT")

        return frozenset(tokens)


# --- Structural Scorer (TF-IDF + structural uniqueness) ---

class StructuralScorer:
    """Scores chunks by TF-IDF content signals + structural fingerprint uniqueness.

    Two independent signals:
    1. Content: standard TF-IDF goal alignment + content uniqueness
       (identical to GoalGuidedScorer)
    2. Structure: Jaccard-based uniqueness of structural fingerprint tokens.
       Chunks with rare structural patterns (action items with line refs,
       IPs, change instructions) score high. Chunks with common structural
       patterns (narrative prose paragraphs) score low.

    Blend: 20% goal alignment + 15% content uniqueness + 25% structural uniqueness
    + 40% structural density (token count / word count, corpus-normalized).
    Structural density dominates because it captures information-dense text
    (needles with line refs, IPs, action markers) vs verbose prose (fillers).
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fingerprinter = StructuralFingerprinter()

    @staticmethod
    def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Score chunks by content TF-IDF + structural uniqueness.

        Args:
            goal: The goal/query message.
            chunks: List of (chunk_hash, content) tuples.
            keyword_scores: Unused, kept for interface compatibility.

        Returns:
            Dict of chunk_hash -> score in [0.5, 2.0].
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not chunks:
            return {}

        n = len(chunks)

        # --- Signal 1 & 2: Content TF-IDF (goal alignment + uniqueness) ---
        texts = [goal] + [content for _, content in chunks]
        try:
            tfidf_matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            return {h: 0.5 for h, _ in chunks}

        goal_vec = tfidf_matrix[0:1]
        chunk_vecs = tfidf_matrix[1:]

        goal_sims = cosine_similarity(goal_vec, chunk_vecs)[0]

        if n > 1:
            pairwise = cosine_similarity(chunk_vecs)
            np.fill_diagonal(pairwise, 0.0)
            avg_peer_sim = pairwise.sum(axis=1) / (n - 1)
            content_uniqueness = 1.0 - avg_peer_sim
        else:
            content_uniqueness = np.array([1.0])

        # --- Signal 3 & 4: Structural fingerprinting ---
        fingerprints = [self._fingerprinter.extract_structural_tokens(content) for _, content in chunks]

        # Signal 3: Structural uniqueness (Jaccard distance to peers)
        structural_uniqueness = np.zeros(n)
        for i in range(n):
            if n > 1:
                peer_sims = []
                for j in range(n):
                    if i != j:
                        peer_sims.append(self._jaccard_similarity(fingerprints[i], fingerprints[j]))
                structural_uniqueness[i] = 1.0 - (sum(peer_sims) / len(peer_sims))
            else:
                structural_uniqueness[i] = 1.0

        # Signal 4: Structural density (structural tokens per word, corpus-normalized)
        # Needles are information-dense (many structural features per word).
        # Fillers are verbose prose (few structural features per word).
        word_counts = [max(len(content.split()), 1) for _, content in chunks]
        raw_densities = [len(fingerprints[i]) / word_counts[i] for i in range(n)]
        max_density = max(raw_densities) if raw_densities else 1.0
        structural_density = np.array([d / max(max_density, 1e-9) for d in raw_densities])

        # --- Blend: 20% goal + 15% content uniqueness + 25% structural uniqueness + 40% density ---
        result: dict[str, float] = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            blended = (
                0.20 * goal_sims[i]
                + 0.15 * content_uniqueness[i]
                + 0.25 * structural_uniqueness[i]
                + 0.40 * structural_density[i]
            )
            score = 0.5 + blended * 1.5
            result[chunk_hash] = max(0.5, min(2.0, score))

        return result


# --- Goal-Guided TF-IDF Scorer ---

class GoalGuidedScorer:
    """Scores chunks by TF-IDF goal similarity + corpus uniqueness.

    Two signals combined:
    1. Goal alignment: cosine similarity to the last user message
    2. Uniqueness: how different a chunk is from all other chunks
       (1 - avg similarity to peers). Needles contain rare specific details
       (line numbers, error names) that score high; filler repeats generic
       themes that score low.

    This handles adversarial cases where keyword overlap is high.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Score chunks by goal similarity + uniqueness.

        Args:
            goal: The goal/query message (last user message).
            chunks: List of (chunk_hash, content) tuples.
            keyword_scores: Optional dict of chunk_hash -> keyword score [0.5, 2.0].

        Returns:
            Dict of chunk_hash -> final_score in [0.5, 2.0].
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not chunks:
            return {}

        # Build corpus: goal first, then all chunks
        texts = [goal] + [content for _, content in chunks]

        try:
            tfidf_matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            return {h: 0.5 for h, _ in chunks}

        goal_vec = tfidf_matrix[0:1]
        chunk_vecs = tfidf_matrix[1:]
        n = chunk_vecs.shape[0]

        # Signal 1: Goal alignment — cosine similarity to goal
        goal_sims = cosine_similarity(goal_vec, chunk_vecs)[0]

        # Signal 2: Uniqueness — 1 - average similarity to all other chunks
        if n > 1:
            pairwise = cosine_similarity(chunk_vecs)
            # Zero out diagonal (self-similarity)
            np.fill_diagonal(pairwise, 0.0)
            avg_peer_sim = pairwise.sum(axis=1) / (n - 1)
            uniqueness = 1.0 - avg_peer_sim
        else:
            uniqueness = np.array([1.0])

        result: dict[str, float] = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            # Blend: 40% goal alignment + 60% uniqueness
            # Uniqueness dominates because during filler turns, goal alignment
            # can't discriminate (goal IS filler), but uniqueness always can.
            blended = 0.4 * goal_sims[i] + 0.6 * uniqueness[i]

            # Map to [0.5, 2.0]
            score = 0.5 + blended * 1.5

            # Clamp
            result[chunk_hash] = max(0.5, min(2.0, score))

        return result


# --- Semantic Scorer (MiniLM embeddings) ---

class SemanticScorer:
    """Scores chunks by MiniLM embedding similarity + semantic distinctiveness.

    Uses all-MiniLM-L6-v2 for dense embeddings. Unlike TF-IDF, this understands
    synonyms, paraphrases, and semantic meaning — a "jwt_validation_error" needle
    can match an "authentication bug" query.

    Also handles boilerplate: two JSON schemas that look identical to TF-IDF
    can have different semantic embeddings if their content differs meaningfully.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Score chunks by semantic goal relevance + distinctiveness.

        Same interface as GoalGuidedScorer for drop-in replacement.
        """
        import numpy as np

        if not chunks:
            return {}

        texts = [goal] + [content for _, content in chunks]
        embeddings = self._model.encode(texts, normalize_embeddings=True)

        goal_emb = embeddings[0:1]
        chunk_embs = embeddings[1:]
        n = chunk_embs.shape[0]

        # Signal 1: Goal relevance — cosine similarity to goal
        # (embeddings are normalized, so dot product = cosine similarity)
        goal_sims = (chunk_embs @ goal_emb.T).flatten()

        # Signal 2: Distinctiveness — 1 - avg cosine similarity to peers
        if n > 1:
            pairwise = chunk_embs @ chunk_embs.T
            np.fill_diagonal(pairwise, 0.0)
            avg_peer_sim = pairwise.sum(axis=1) / (n - 1)
            distinctiveness = 1.0 - avg_peer_sim
        else:
            distinctiveness = np.array([1.0])

        result: dict[str, float] = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            blended = 0.4 * goal_sims[i] + 0.6 * distinctiveness[i]
            score = 0.5 + blended * 1.5
            result[chunk_hash] = max(0.5, min(2.0, score))

        return result


# --- Entity-Aware Scorer (TF-IDF + thresholded entity bonus) ---

class EntityAwareScorer:
    """Scores chunks by standard TF-IDF + thresholded entity overlap bonus.

    Uses the same 40/60 goal/uniqueness blend as GoalGuidedScorer, then adds
    a small entity bonus ONLY when entity overlap with the goal exceeds 20%.
    This threshold prevents random entity matches during filler turns from
    disrupting TF-IDF scoring, while allowing strong entity matches (e.g.,
    needle entities matching a recall question) to provide a boost.
    """

    _ENTITY_THRESHOLD = 0.4  # Minimum entity overlap to trigger bonus
    _ENTITY_BONUS_WEIGHT = 0.5  # Max bonus at 100% overlap

    def __init__(self):
        self._tfidf = GoalGuidedScorer()
        self._extractor = EntityExtractor()

    @staticmethod
    def _expand_entities(entities: set[str]) -> set[str]:
        """Expand entity set with underscore/space/hyphen variants."""
        expanded = set(entities)
        for e in entities:
            if '_' in e:
                expanded.add(e.replace('_', ' '))
                expanded.add(e.replace('_', '-'))
            if '-' in e:
                expanded.add(e.replace('-', '_'))
                expanded.add(e.replace('-', ' '))
        return expanded

    def _entity_overlap(self, chunk_entities: set[str], goal_entities_expanded: set[str]) -> float:
        """Compute entity overlap score: fraction of chunk entities matching goal."""
        if not chunk_entities or not goal_entities_expanded:
            return 0.0
        expanded_chunk = self._expand_entities(chunk_entities)
        matches = len(expanded_chunk & goal_entities_expanded)
        return min(1.0, matches / len(chunk_entities))

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not chunks:
            return {}

        # Standard TF-IDF signals (identical to GoalGuidedScorer)
        texts = [goal] + [content for _, content in chunks]
        try:
            tfidf_matrix = self._tfidf._vectorizer.fit_transform(texts)
        except ValueError:
            return {h: 0.5 for h, _ in chunks}

        goal_vec = tfidf_matrix[0:1]
        chunk_vecs = tfidf_matrix[1:]
        n = chunk_vecs.shape[0]

        goal_sims = cosine_similarity(goal_vec, chunk_vecs)[0]

        if n > 1:
            pairwise = cosine_similarity(chunk_vecs)
            np.fill_diagonal(pairwise, 0.0)
            avg_peer_sim = pairwise.sum(axis=1) / (n - 1)
            uniqueness = 1.0 - avg_peer_sim
        else:
            uniqueness = np.array([1.0])

        # Entity overlap (thresholded + specificity check)
        goal_entities = self._extractor.extract_entities(goal)
        goal_entities_expanded = self._expand_entities(goal_entities)

        # Compute raw entity overlaps
        raw_overlaps = []
        for _, content in chunks:
            chunk_entities = self._extractor.extract_entities(content)
            raw_overlaps.append(self._entity_overlap(chunk_entities, goal_entities_expanded))

        # Specificity check: if the goal matches too many chunks above threshold,
        # it's a generic/filler message — disable entity bonus entirely.
        # This prevents adversarial fillers (which share entities) from boosting each other.
        matching_count = sum(1 for o in raw_overlaps if o >= self._ENTITY_THRESHOLD)
        apply_bonus = matching_count <= max(3, n // 10)  # At most ~10% of chunks

        entity_scores = []
        for overlap in raw_overlaps:
            if apply_bonus and overlap >= self._ENTITY_THRESHOLD:
                entity_scores.append((overlap - self._ENTITY_THRESHOLD) * self._ENTITY_BONUS_WEIGHT)
            else:
                entity_scores.append(0.0)

        result: dict[str, float] = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            # Standard TF-IDF base
            tfidf_blend = 0.4 * goal_sims[i] + 0.6 * uniqueness[i]
            # Entity bonus (only fires above threshold)
            combined = tfidf_blend + entity_scores[i]
            score = 0.5 + combined * 1.5
            result[chunk_hash] = max(0.5, min(2.0, score))

        return result


class BM25Scorer:
    """BM25-based scorer using rank_bm25 library.

    40% goal relevance + 60% corpus uniqueness (1 - avg peer similarity).
    """

    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "am", "it", "its",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "about", "up", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "also",
    })

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r'[a-z0-9_]+', text.lower())
        return [t for t in tokens if len(t) > 1 and t not in self.STOP_WORDS]

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict | None = None
    ) -> dict[str, float]:
        if not chunks:
            return {}
        from rank_bm25 import BM25Okapi

        goal_tokens = self._tokenize(goal)
        if not goal_tokens:
            return {h: 0.5 for h, _ in chunks}

        corpus = [self._tokenize(text) for _, text in chunks]
        if not any(corpus):
            return {h: 0.5 for h, _ in chunks}

        bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        raw = bm25.get_scores(goal_tokens)

        max_raw = max(raw) if max(raw) > 0 else 1.0
        scores = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            goal_score = raw[i] / max_raw  # 0..1
            # Uniqueness: 1 - avg similarity to other chunks
            if len(chunks) > 1:
                other_scores = []
                doc_tokens = corpus[i]
                if doc_tokens:
                    for j, other_tokens in enumerate(corpus):
                        if i != j and other_tokens:
                            intersection = set(doc_tokens) & set(other_tokens)
                            union = set(doc_tokens) | set(other_tokens)
                            other_scores.append(len(intersection) / len(union) if union else 0)
                uniqueness = 1.0 - (sum(other_scores) / len(other_scores) if other_scores else 0)
            else:
                uniqueness = 1.0
            blended = 0.4 * goal_score + 0.6 * uniqueness
            # Map to [0.5, 2.0]
            scores[chunk_hash] = 0.5 + blended * 1.5
        return scores


@dataclass(frozen=True)
class DecisionRecord:
    timestamp: float
    action: str  # "append", "compact_soft", "compact_hard", "drop"
    chunk_hash: str
    reason: str
    context_size_before: int
    context_size_after: int


@dataclass
class ChunkEntry:
    chunk_hash: str
    role: str
    content: str
    tokens: int
    turn: int
    priority: float
    timestamp: float


class ChunkLog:
    """Append-only context log with soft/hard threshold compaction.

    Args:
        db_path: SQLite database path (":memory:" for in-memory).
        max_tokens: Maximum context window size in tokens.
        soft_threshold: Ratio of max_tokens that triggers soft compaction (summarize old).
        hard_threshold: Ratio of max_tokens that triggers hard compaction (drop low-priority).
            Set both to 2.0 to effectively disable compaction.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_tokens: int = 128_000,
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        auto_priority: bool = False,
        goal_guided: bool = False,
        scoring_mode: str | None = "structural",
    ):
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.auto_priority = auto_priority
        self.goal_guided = goal_guided
        self.scoring_mode = scoring_mode  # 'structural' (default), 'bm25', 'openhands', 'tfidf', 'semantic', 'hybrid', 'entity_aware', or None
        self._turn = 0
        self._compaction_count = 0
        self._decisions: list[DecisionRecord] = []
        self._last_user_message: str = ""
        self._accumulated_keywords: set[str] = set()
        # Initialize scorers based on mode
        self._entity_scorer: EntityAwareScorer | None = None
        self._structural_scorer: StructuralScorer | None = None
        self._bm25_scorer = None
        if scoring_mode == "openhands":
            # OpenHands-style: no content scorer, purely positional (U-shaped by turn)
            self._goal_scorer = None
            self._semantic_scorer = None
            self._openhands_keep_first = 1  # preserve first K chunks
        elif scoring_mode == "bm25":
            self._bm25_scorer = BM25Scorer()
            self._goal_scorer = None
            self._semantic_scorer = None
            self._structural_scorer = None
        elif scoring_mode == "structural":
            self._goal_scorer: GoalGuidedScorer | None = None
            self._semantic_scorer: SemanticScorer | None = None
            self._structural_scorer = StructuralScorer()
        elif scoring_mode == "entity_aware":
            self._goal_scorer = None
            self._semantic_scorer = None
            self._entity_scorer = EntityAwareScorer()
        elif scoring_mode == "semantic":
            self._goal_scorer = None
            self._semantic_scorer = SemanticScorer()
        elif scoring_mode == "hybrid":
            self._goal_scorer = GoalGuidedScorer()
            self._semantic_scorer = SemanticScorer()
        elif goal_guided or scoring_mode == "tfidf":
            self._goal_scorer = GoalGuidedScorer()
            self._semantic_scorer = None
        else:
            self._goal_scorer = None
            self._semantic_scorer = None

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_hash TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                priority REAL NOT NULL DEFAULT 1.0,
                timestamp REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                context_size_before INTEGER NOT NULL,
                context_size_after INTEGER NOT NULL
            )"""
        )
        self._conn.commit()

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def decisions(self) -> list[DecisionRecord]:
        return list(self._decisions)

    def current_tokens(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM chunks").fetchone()
        return row[0]

    def turn(self) -> int:
        return self._turn

    def append(self, role: str, content: str, priority: float = 1.0) -> str:
        """Append content to the log. Returns the chunk hash."""
        chunk_hash = _sha256(f"{role}:{content}")
        tokens = _estimate_tokens(content)
        now = time.time()

        # Track last user message and accumulate keywords for scoring modes
        if role == "user":
            self._last_user_message = content
            if self.auto_priority or self.goal_guided or self.scoring_mode:
                self._accumulated_keywords.update(extract_keywords(content))

        # Content-addressed: skip if already present
        existing = self._conn.execute(
            "SELECT chunk_hash FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
        ).fetchone()
        if existing:
            return chunk_hash

        size_before = self.current_tokens()
        self._conn.execute(
            "INSERT INTO chunks (chunk_hash, role, content, tokens, turn, priority, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk_hash, role, content, tokens, self._turn, priority, now),
        )
        self._conn.commit()

        self._record_decision("append", chunk_hash, f"role={role} tokens={tokens}", size_before, size_before + tokens)

        # Check compaction thresholds
        self._maybe_compact()
        return chunk_hash

    def next_turn(self) -> None:
        self._turn += 1

    def get_context(self) -> list[dict[str, str]]:
        """Get current context as a list of messages, ordered by turn then timestamp."""
        rows = self._conn.execute(
            "SELECT role, content FROM chunks ORDER BY turn ASC, timestamp ASC"
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def get_context_tokens(self) -> int:
        return self.current_tokens()

    def _rescore_chunks_auto(self) -> None:
        """Re-score all chunks based on accumulated keywords from user messages."""
        if not self._accumulated_keywords:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        for chunk_hash, content in rows:
            new_priority = score_chunk(content, self._accumulated_keywords)
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_goal_guided(self) -> None:
        """Re-score all chunks using TF-IDF similarity to last user message."""
        if not self._last_user_message or not self._goal_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        # Pure TF-IDF scoring — subsumes keyword matching and handles
        # adversarial cases where keyword overlap makes keyword scores useless
        scores = self._goal_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )

        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_semantic(self) -> None:
        """Re-score all chunks using MiniLM semantic embeddings."""
        if not self._last_user_message or not self._semantic_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        scores = self._semantic_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )

        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_structural(self) -> None:
        """Re-score all chunks using TF-IDF + structural fingerprinting."""
        if not self._last_user_message or not self._structural_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        scores = self._structural_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )

        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_openhands(self) -> None:
        """Re-score chunks using OpenHands-style positional (U-shaped) priorities.

        Implements amortized forgetting: first K chunks and most recent chunks
        get high scores (2.0), middle chunks get low scores (0.5).
        This causes the compaction logic to evict the middle, preserving
        head (initial context) and tail (recent context).
        """
        keep_first = getattr(self, '_openhands_keep_first', 1)
        rows = self._conn.execute(
            "SELECT chunk_hash, turn, tokens FROM chunks ORDER BY turn ASC, rowid ASC"
        ).fetchall()
        if not rows:
            return

        n = len(rows)
        # Calculate how many tail chunks to protect based on token budget
        # Target: keep enough tail chunks to fill ~50% of max_tokens (halving strategy)
        target_tokens = int(self.max_tokens * self.soft_threshold * 0.5)
        head_tokens = sum(t for _, _, t in rows[:keep_first])
        tail_budget = target_tokens - head_tokens

        # Count tail chunks from end
        tail_tokens = 0
        tail_start = n
        for i in range(n - 1, keep_first - 1, -1):
            if tail_tokens + rows[i][2] <= tail_budget:
                tail_tokens += rows[i][2]
                tail_start = i
            else:
                break

        for i, (chunk_hash, turn, tokens) in enumerate(rows):
            if i < keep_first:
                priority = 2.0  # Head: always preserve
            elif i >= tail_start:
                priority = 2.0  # Tail: always preserve
            else:
                priority = 0.5  # Middle: expendable
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_bm25(self) -> None:
        """Re-score all chunks using BM25."""
        if not self._last_user_message or not self._bm25_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return
        scores = self._bm25_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )
        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_entity_aware(self) -> None:
        """Re-score all chunks using TF-IDF + entity matching."""
        if not self._last_user_message or not self._entity_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        scores = self._entity_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )

        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_hybrid(self) -> None:
        """Re-score using 30% TF-IDF + 70% semantic blend."""
        if not self._last_user_message:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        tfidf_scores = {}
        semantic_scores = {}
        if self._goal_scorer:
            tfidf_scores = self._goal_scorer.score_chunks(
                self._last_user_message, rows, keyword_scores=None
            )
        if self._semantic_scorer:
            semantic_scores = self._semantic_scorer.score_chunks(
                self._last_user_message, rows, keyword_scores=None
            )

        for chunk_hash, _ in rows:
            t = tfidf_scores.get(chunk_hash, 0.5)
            s = semantic_scores.get(chunk_hash, 0.5)
            blended = 0.3 * t + 0.7 * s
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (blended, chunk_hash),
            )
        self._conn.commit()

    def _maybe_compact(self) -> None:
        # Thresholds > 1.0 mean "never compact"
        if self.soft_threshold > 1.0 and self.hard_threshold > 1.0:
            return
        current = self.current_tokens()
        hard_limit = int(self.max_tokens * self.hard_threshold)
        soft_limit = int(self.max_tokens * self.soft_threshold)

        if current > hard_limit or current > soft_limit:
            # Re-score chunks before compaction
            if self.scoring_mode == "openhands":
                self._rescore_chunks_openhands()
            elif self.scoring_mode == "bm25":
                self._rescore_chunks_bm25()
            elif self.scoring_mode == "structural":
                self._rescore_chunks_structural()
            elif self.scoring_mode == "entity_aware":
                self._rescore_chunks_entity_aware()
            elif self.scoring_mode == "semantic":
                self._rescore_chunks_semantic()
            elif self.scoring_mode == "hybrid":
                self._rescore_chunks_hybrid()
            elif self.scoring_mode == "tfidf" or self.goal_guided:
                self._rescore_chunks_goal_guided()
            elif self.auto_priority:
                self._rescore_chunks_auto()

        if current > hard_limit:
            self._compact_hard(current)
        elif current > soft_limit:
            self._compact_soft(current)

    def _compact_soft(self, current_tokens: int) -> None:
        """Soft compaction: drop oldest low-priority chunks."""
        self._compaction_count += 1
        target = int(self.max_tokens * self.soft_threshold * 0.8)
        rows = self._conn.execute(
            "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
        ).fetchall()

        removed = 0
        use_scoring = self.auto_priority or self.goal_guided or self.scoring_mode in ("bm25", "openhands", "tfidf", "semantic", "hybrid", "entity_aware", "structural")
        for chunk_hash, tokens, priority, turn in rows:
            if current_tokens - removed <= target:
                break
            # With any scoring mode, trust the scoring — evict any low-priority chunk
            if use_scoring:
                if priority < 1.5:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_soft", chunk_hash,
                        f"priority={priority:.2f} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
            else:
                if priority < 1.5 and turn < self._turn - 1:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_soft", chunk_hash,
                        f"priority={priority} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
        self._conn.commit()

    def _compact_hard(self, current_tokens: int) -> None:
        """Hard compaction: aggressively drop to get below threshold."""
        self._compaction_count += 1
        target = int(self.max_tokens * self.soft_threshold * 0.6)
        rows = self._conn.execute(
            "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
        ).fetchall()

        removed = 0
        use_scoring = self.auto_priority or self.goal_guided or self.scoring_mode in ("bm25", "openhands", "tfidf", "semantic", "hybrid", "entity_aware", "structural")
        # First pass: try to evict only low-priority chunks (scoring protection)
        if use_scoring:
            for chunk_hash, tokens, priority, turn in rows:
                if current_tokens - removed <= target:
                    break
                if priority >= 1.5:
                    continue
                if turn < self._turn:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_hard", chunk_hash,
                        f"priority={priority:.2f} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
            # If still over hard threshold, fall back to standard eviction (no priority floor)
            if current_tokens - removed > int(self.max_tokens * self.hard_threshold):
                rows = self._conn.execute(
                    "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
                ).fetchall()
                for chunk_hash, tokens, priority, turn in rows:
                    if current_tokens - removed <= target:
                        break
                    if turn < self._turn:
                        self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                        removed += tokens
                        self._record_decision(
                            "compact_hard", chunk_hash,
                            f"priority={priority:.2f} turn={turn} (fallback)",
                            current_tokens - removed + tokens, current_tokens - removed,
                        )
        else:
            for chunk_hash, tokens, priority, turn in rows:
                if current_tokens - removed <= target:
                    break
                if turn < self._turn:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_hard", chunk_hash,
                        f"priority={priority} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
        self._conn.commit()

    def _record_decision(self, action: str, chunk_hash: str, reason: str, before: int, after: int) -> None:
        now = time.time()
        rec = DecisionRecord(
            timestamp=now, action=action, chunk_hash=chunk_hash,
            reason=reason, context_size_before=before, context_size_after=after,
        )
        self._decisions.append(rec)
        self._conn.execute(
            "INSERT INTO decisions (timestamp, action, chunk_hash, reason, context_size_before, context_size_after) VALUES (?, ?, ?, ?, ?, ?)",
            (now, action, chunk_hash, reason, before, after),
        )

    def close(self) -> None:
        self._conn.close()


class CerebrasSession:
    """Wrapper around Cerebras API with context management via ChunkLog.

    Tracks TTFT, token usage, and handles rate limiting with exponential backoff.
    """

    def __init__(
        self,
        chunk_log: ChunkLog,
        model: str = "llama3.1-8b",
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        from cerebras.cloud.sdk import Cerebras

        self.chunk_log = chunk_log
        self.model = model
        self.max_retries = max_retries
        self._client = Cerebras(api_key=api_key or os.environ.get("CEREBRAS_API_KEY"))
        self._ttft_samples: list[float] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_turns = 0

    @property
    def avg_ttft(self) -> float:
        if not self._ttft_samples:
            return 0.0
        return sum(self._ttft_samples) / len(self._ttft_samples)

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def chat(self, system_prompt: str | None = None, max_completion_tokens: int | None = None) -> str:
        """Send current context to Cerebras and return the response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chunk_log.get_context())

        # Calculate available tokens for completion with safety margin
        input_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        model_limit = self.chunk_log.max_tokens
        safety_margin = 512  # reserve buffer for tokenizer estimation drift
        available = max(256, model_limit - input_tokens - safety_margin)
        completion_tokens = min(max_completion_tokens or 16384, available)

        response_text = ""
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=completion_tokens,
                    temperature=0.0,
                )
                ttft = time.time() - t0
                self._ttft_samples.append(ttft)

                choice = response.choices[0]
                response_text = choice.message.content or ""
                # For reasoning models: if content is empty, include reasoning as fallback
                if not response_text and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    response_text = choice.message.reasoning

                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens
                    self._total_output_tokens += response.usage.completion_tokens

                self._total_turns += 1
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Cerebras API failed after {self.max_retries} retries: {e}") from e

        # Append assistant response to context
        self.chunk_log.append("assistant", response_text, priority=1.0)
        self.chunk_log.next_turn()
        return response_text

    def get_metrics(self) -> dict[str, Any]:
        return {
            "avg_ttft": round(self.avg_ttft, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_turns": self.total_turns,
            "context_size_tokens": self.chunk_log.get_context_tokens(),
            "compaction_events": self.chunk_log.compaction_count,
        }
