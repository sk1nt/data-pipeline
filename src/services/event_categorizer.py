"""Event categorizer — classifies a social/news text into one of a fixed set of
market-relevant categories and extracts the "first mention" source fingerprint.

Categories drive the similarity bucketing in ProjectionService: when a new alert
fires we look up past events in the *same category* to estimate likely move size.

First-mention detection
-----------------------
Given a piece of text (headline, tweet) the categorizer:
1. Normalises it to a fingerprint (lower-case, strip punctuation, collapse spaces).
2. Queries the correlation_events DuckDB table for the earliest stored event whose
   fingerprint is within edit-distance 2 of the new one (near-duplicate detection).
3. If the new event is the earliest, it's a "primary source" for that story.
4. The source_rank field rates the author/outlet's historical move-impact to weight
   the alert priority (populated by SourceCredibilityService).

Source types detected
---------------------
  institution   — Fed, Treasury, IMF, ECB, BIS, SEC, CFTC, DoJ, DoD, White House
  executive     — named CEOs/CFOs/founders (e.g. "musk", "powell", "yellen")
  media         — Bloomberg, Reuters, WSJ, FT, CNBC, Fox, AP, Axios
  government    — presidents, senators, government departments
  social        — Truth Social / Twitter individuals without institutional affiliation
  unknown       — anything else
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

# Each entry: (category_name, list_of_trigger_patterns, base_severity)
# Patterns are matched case-insensitively against the full event text.
# More specific patterns should come first (first match wins within a tier).

_CATEGORY_RULES: List[Tuple[str, List[str], str]] = [
    # ── Tier 1: highest market impact ────────────────────────────────────
    ("fomc_rate_decision", [
        r"fomc", r"rate decision", r"interest rate decision",
        r"fed raises", r"fed cuts", r"rate hike", r"rate cut",
        r"federal reserve.*decision", r"fed funds rate",
    ], "critical"),

    ("fed_speech", [
        r"\bpowell\b", r"\bwaller\b", r"\bjefferson\b", r"\bcook\b",
        r"federal reserve.*chair", r"fed chair", r"fed governor",
        r"fed speak", r"fed comment", r"fed official",
        r"dovish", r"hawkish", r"fed pivot",
    ], "high"),

    ("cpi_inflation", [
        r"\bcpi\b", r"consumer price index", r"core cpi",
        r"\bpce\b", r"personal consumption", r"core pce",
        r"\bppi\b", r"producer price", r"inflation report",
        r"inflation data", r"inflation surged", r"inflation fell",
    ], "high"),

    ("jobs_labor", [
        r"\bnfp\b", r"nonfarm payroll", r"non-?farm payroll",
        r"unemployment rate", r"jobless claims", r"initial claims",
        r"continuing claims", r"labor market", r"jobs report",
        r"jobs added", r"employment situation",
    ], "high"),

    ("gdp", [
        r"\bgdp\b", r"gross domestic product", r"gdp growth",
        r"gdp shrank", r"economic growth", r"recession confirmed",
    ], "high"),

    ("tariff_trade", [
        r"\btariff", r"trade war", r"trade deal", r"import tax",
        r"trade barrier", r"trade representative", r"trade deficit",
        r"trade surplus", r"trade agreement", r"customs duty",
        r"embargo", r"export ban", r"trade sanction",
    ], "high"),

    ("geopolitical_escalation", [
        r"airstrike", r"missile strike", r"nuclear", r"declaration of war",
        r"military conflict", r"troops deployed", r"invasion",
        r"escalat", r"retaliat", r"ceasefire broken", r"hostilities",
        r"nato invok", r"article 5",
    ], "high"),

    # ── Tier 2: significant but more routine ─────────────────────────────
    ("earnings_major", [
        r"earnings.*beat", r"earnings.*miss", r"earnings.*surprise",
        r"eps beat", r"eps miss", r"revenue beat", r"revenue miss",
        r"quarterly results", r"q[1-4] earnings", r"full.?year guidance",
    ], "medium"),

    ("ism_pmi", [
        r"\bism\b", r"ism manufacturing", r"ism services",
        r"\bpmi\b", r"purchasing managers", r"manufacturing index",
        r"services index",
    ], "medium"),

    ("retail_consumer", [
        r"retail sales", r"consumer confidence", r"consumer sentiment",
        r"consumer spending", r"personal spending", r"michigan sentiment",
        r"conference board",
    ], "medium"),

    ("housing", [
        r"housing starts", r"building permits", r"existing home sales",
        r"new home sales", r"case-shiller", r"pending home sales",
        r"mortgage rate",
    ], "medium"),

    ("durable_goods", [
        r"durable goods", r"factory orders", r"capital goods",
        r"core capital goods",
    ], "medium"),

    ("geopolitical_tension", [
        r"\bchina\b", r"\brussia\b", r"\bukraine\b", r"\biran\b",
        r"\bnorth korea\b", r"middle east", r"south china sea",
        r"taiwan strait", r"nato", r"\bopec\b", r"energy sanction",
    ], "medium"),

    ("debt_fiscal", [
        r"debt ceiling", r"government shutdown", r"continuing resolution",
        r"budget deal", r"spending bill", r"deficit", r"national debt",
        r"treasury auction", r"bond yield", r"10.?year yield",
    ], "medium"),

    # ── Tier 3: lower signal, still logged ───────────────────────────────
    ("crypto", [
        r"\bbitcoin\b", r"\bcrypto\b", r"\bethereum\b", r"\bbtc\b",
        r"cryptocurrency", r"stablecoin", r"defi", r"nft",
    ], "low"),

    ("mag7_company", [
        r"\bapple\b", r"\baapl\b", r"\bmicrosoft\b", r"\bmsft\b",
        r"\bgoogle\b", r"\bgoogl\b", r"\bamazon\b", r"\bamzn\b",
        r"\bmeta\b", r"\bnvidia\b", r"\bnvda\b", r"\btesla\b",
        r"\btsla\b", r"\balphanet\b",
    ], "low"),

    ("ai_tech", [
        r"\bartificial intelligence\b", r"\b(open)?ai\b", r"\bchatgpt\b",
        r"\bgpt-?[0-9]", r"\bllm\b", r"large language model",
        r"ai regulation", r"ai chip", r"ai investment",
    ], "low"),

    ("general_market", [
        r"stock market", r"wall street", r"\bnasdaq\b", r"\bdow jones\b",
        r"\bs&p 500\b", r"market rally", r"market sell.?off",
        r"market correction", r"risk.?off", r"risk.?on",
    ], "low"),
]

# Compiled at module load
_COMPILED_RULES: List[Tuple[str, List[re.Pattern], str]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns], sev)
    for name, patterns, sev in _CATEGORY_RULES
]


# ---------------------------------------------------------------------------
# Source type classification
# ---------------------------------------------------------------------------

_INSTITUTION_AUTHORS = {
    "federal reserve", "fed", "fomc", "treasury", "imf", "world bank",
    "ecb", "boe", "bis", "sec", "cftc", "doj", "dod", "white house",
    "state department", "oecd", "fdic", "occ", "cfpb",
    "opec", "iea", "wto", "nato",
}

_EXECUTIVE_NAMES = {
    "powell", "yellen", "waller", "jefferson", "cook", "mester",
    "musk", "cook", "bezos", "zuckerberg", "altman", "huang",
    "dimon", "buffett", "dalio", "ackman", "icahn",
    "trump", "biden", "bessent", "lutnick",
}

_MEDIA_AUTHORS = {
    "bloomberg", "reuters", "wsj", "wall street journal", "financial times",
    "ft", "cnbc", "fox business", "barrons", "marketwatch", "axios",
    "ap", "associated press", "the economist", "nytimes", "new york times",
    "politico", "semafor", "business insider", "yahoo finance",
    "seeking alpha", "zerohedge",
}

_GOVT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"senator\b", r"congressman\b", r"representative\b",
        r"white house\b", r"department of", r"treasury secretary",
        r"commerce secretary", r"labor secretary", r"fed chair",
    ]
]


def classify_source_type(author: str, text: str = "") -> str:
    """Return one of: institution | executive | media | government | social | unknown."""
    a = author.lower().strip()
    combined = (a + " " + text.lower())[:300]

    if a in _INSTITUTION_AUTHORS or any(inst in a for inst in _INSTITUTION_AUTHORS):
        return "institution"
    if a in _EXECUTIVE_NAMES or any(name in a for name in _EXECUTIVE_NAMES):
        return "executive"
    if a in _MEDIA_AUTHORS or any(m in a for m in _MEDIA_AUTHORS):
        return "media"
    if any(p.search(combined) for p in _GOVT_PATTERNS):
        return "government"
    # Truth Social / Twitter — individual with no institutional match
    if author and author not in ("unknown", ""):
        return "social"
    return "unknown"


# ---------------------------------------------------------------------------
# Text fingerprinting for first-mention detection
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "or", "but", "in",
    "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
    "into", "through", "as", "its", "it", "this", "that", "these", "those",
    "says", "said", "sources", "report", "reports", "new", "breaking",
])


def text_fingerprint(text: str) -> str:
    """Normalise text to a compact fingerprint for near-duplicate detection.

    Steps: unicode NFC → lower → strip punctuation → remove stopwords →
    sort tokens alphabetically → join with space (order-independent match).
    """
    # NFC normalise
    text = unicodedata.normalize("NFC", text)
    # Lower
    text = text.lower()
    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # Strip non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Tokenise and remove stopwords / short tokens
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
    # Sort for order-independence
    tokens.sort()
    return " ".join(tokens[:30])  # cap at 30 tokens


def fingerprint_similarity(a: str, b: str) -> float:
    """Jaccard similarity of token sets (0.0–1.0).  >= 0.6 considered near-duplicate."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Main categorizer dataclass
# ---------------------------------------------------------------------------

@dataclass
class EventClassification:
    """Result of classifying one event text."""
    category: str = "uncategorized"
    severity: str = "low"           # critical | high | medium | low
    source_type: str = "unknown"    # institution | executive | media | government | social | unknown
    matched_patterns: List[str] = field(default_factory=list)
    fingerprint: str = ""
    # Populated by caller after DB lookup:
    is_first_mention: Optional[bool] = None   # None = not yet checked
    first_mention_id: Optional[str] = None    # alert_id of the earliest matching event


class EventCategorizer:
    """Classify event text into a market category and source type."""

    def classify(self, text: str, author: str = "", source: str = "") -> EventClassification:
        """Return an EventClassification for the given text + author."""
        category = "uncategorized"
        severity = "low"
        matched: List[str] = []

        for name, patterns, sev in _COMPILED_RULES:
            hits = [p.pattern for p in patterns if p.search(text)]
            if hits:
                category = name
                severity = sev
                matched = hits
                break  # first match wins (rules ordered by priority)

        source_type = classify_source_type(author, text)
        fp = text_fingerprint(text)

        return EventClassification(
            category=category,
            severity=severity,
            source_type=source_type,
            matched_patterns=matched[:5],  # store up to 5 for debugging
            fingerprint=fp,
        )

    def classify_batch(
        self, events: List[Dict]
    ) -> List[Tuple[Dict, EventClassification]]:
        """Classify a list of event dicts (must have 'social_text' and 'social_author')."""
        results = []
        seen_fps: Dict[str, str] = {}  # fingerprint → first event_id seen (within this batch)

        for ev in events:
            text = ev.get("social_text") or ev.get("text", "")
            author = ev.get("social_author") or ev.get("author", "")
            cls = self.classify(text, author)

            # Within-batch first-mention: earliest timestamp in the batch is the primary
            fp = cls.fingerprint
            ev_id = ev.get("social_event_id") or ev.get("event_id", "")
            if fp and fp not in seen_fps:
                seen_fps[fp] = ev_id
                cls.is_first_mention = True
                cls.first_mention_id = ev_id
            elif fp in seen_fps:
                cls.is_first_mention = False
                cls.first_mention_id = seen_fps[fp]

            results.append((ev, cls))
        return results
