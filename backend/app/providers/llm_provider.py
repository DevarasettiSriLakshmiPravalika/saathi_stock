"""
LLM provider abstraction for claim extraction and query intent parsing.

CRITICAL RULES (Contract §53):
- LLM extracts structured data ONLY.
- LLM NEVER directly modifies inventory.
- LLM NEVER approves or rejects statements.
- All LLM output is validated by Pydantic before use.
- Invalid LLM output is safely rejected.
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word-to-number conversion table (English, 0-999 range needed in practice)
# ---------------------------------------------------------------------------
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE = {"hundred": 100, "thousand": 1000}

_WORD_NUM_PATTERN = re.compile(
    r"\b(" + "|".join(list(_ONES) + list(_TENS) + list(_SCALE)) + r")\b",
    re.IGNORECASE,
)


def _words_to_number(text: str) -> Optional[float]:
    """
    Convert a short English number phrase to a float.
    Handles: one, twenty, forty two, three hundred, etc.
    Returns None if no word-number phrase is found.
    """
    words = text.lower().split()
    result = 0
    current = 0
    found = False
    for word in words:
        word = word.strip(",.-")
        if word in _ONES:
            current += _ONES[word]
            found = True
        elif word in _TENS:
            current += _TENS[word]
            found = True
        elif word == "hundred":
            if current == 0:
                current = 1
            current *= 100
            found = True
        elif word == "thousand":
            if current == 0:
                current = 1
            result += current * 1000
            current = 0
            found = True
    result += current
    return float(result) if found else None


def _extract_quantity(text: str) -> float:
    """
    Extract a numeric quantity from text.
    Tries digit parsing first, then English word-number conversion.
    Falls back to 1.0.
    """
    # Digit-first: plain integer or decimal
    digit_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if digit_match:
        return float(digit_match.group(1))

    # Word-number fallback
    word_num = _words_to_number(text)
    if word_num is not None and word_num > 0:
        return word_num

    return 1.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ExtractedClaim(BaseModel):
    """
    Structured claim extracted from transcript.
    All fields validated — invalid LLM output raises ValidationError.
    """
    product: str
    quantity: float
    unit: str
    direction: str  # IN or OUT
    actor: Optional[str] = "self"
    event_time: Optional[str] = None

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.upper()
        if v not in ("IN", "OUT"):
            raise ValueError(f"direction must be IN or OUT, got: {v}")
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v


class QueryIntent(BaseModel):
    """Parsed intent from owner natural-language query."""
    intent: str  # CURRENT_STOCK | SALES_TODAY | RECEIPTS_TODAY | etc.
    product_name: Optional[str] = None
    date_filter: Optional[str] = None
    raw_query: str


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @abstractmethod
    async def extract_claim(self, transcript: str) -> Optional[ExtractedClaim]:
        """Extract structured claim from transcript. Returns None if extraction fails."""
        ...

    @abstractmethod
    async def parse_query_intent(self, query: str) -> QueryIntent:
        """Parse owner natural-language query into structured intent."""
        ...

    @abstractmethod
    async def generate_explanation(self, context: dict) -> str:
        """Generate human-readable explanation based on actual backend evidence."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Mock / rule-based provider (kept for deterministic testing)
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """
    Development LLM provider — uses rule-based parsing.
    Returns deterministic results for testing.
    Used when SAATHI_LLM_PROVIDER=mock.

    Handles both digit quantities (5, 20) and English number words
    (five, twenty, forty, etc.).
    """

    @property
    def provider_name(self) -> str:
        return "mock-llm-v1"

    async def extract_claim(self, transcript: str) -> Optional[ExtractedClaim]:
        """
        Rule-based claim extraction for development/testing.
        Parses patterns like: 'Sold 5 bags of rice', 'Received twenty boxes of oil'.
        """
        t = transcript.lower().strip()

        # ── Direction ────────────────────────────────────────────────────────
        out_keywords = [
            "sold", "sell", "gave", "dispatched", "sent", "sale",
            "delivered to customer", "delivered",
        ]
        in_keywords = [
            "received", "got", "added", "purchased", "bought", "arrived",
            "delivered from supplier", "inward", "restocked", "stock in",
        ]

        direction: Optional[str] = None
        for kw in out_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                direction = "OUT"
                break
        if not direction:
            for kw in in_keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", t):
                    direction = "IN"
                    break
        if not direction:
            direction = "IN"

        # ── Quantity — digits first, then word-numbers ────────────────────────
        quantity = _extract_quantity(t)

        # ── Unit ─────────────────────────────────────────────────────────────
        unit_map = {
            "bags": "bag", "bag": "bag",
            "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg",
            "liter": "liter", "liters": "liter", "litre": "liter", "litres": "liter",
            "packet": "packet", "packets": "packet",
            "piece": "piece", "pieces": "piece",
            "box": "box", "boxes": "box",
            "ton": "ton", "tons": "ton",
            "quintal": "quintal", "quintals": "quintal",
            "carton": "carton", "cartons": "carton",
            "unit": "unit", "units": "unit",
        }
        unit = "unit"
        # Match longest token first to avoid partial matches
        for raw_unit in sorted(unit_map.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(raw_unit) + r"\b", t):
                unit = unit_map[raw_unit]
                break

        # ── Product — strip everything known to find the noun ─────────────────
        clean = re.sub(r"[^\w\s]", " ", t)
        # Remove digit quantities
        clean = re.sub(r"\b\d+(?:\.\d+)?\b", " ", clean)
        # Remove word-numbers
        clean = _WORD_NUM_PATTERN.sub(" ", clean)
        # Remove units
        for raw_unit in unit_map:
            clean = re.sub(r"\b" + re.escape(raw_unit) + r"\b", " ", clean)
        # Remove direction/function words
        all_stopwords = (
            out_keywords + in_keywords
            + ["of", "the", "a", "an", "from", "to", "supplier", "customer", "please", "out", "in"]
        )
        for kw in all_stopwords:
            clean = re.sub(r"\b" + re.escape(kw) + r"\b", " ", clean)
        words = [w for w in clean.split() if len(w) >= 2]
        product = " ".join(words[:2]) if words else "unknown"

        if not product or product.lower() == "unknown":
            return None

        try:
            return ExtractedClaim(
                product=product.strip().title(),
                quantity=quantity,
                unit=unit,
                direction=direction,
            )
        except Exception:
            return None

    async def parse_query_intent(self, query: str) -> QueryIntent:
        """Rule-based query intent parsing."""
        q = query.lower().strip()

        if any(w in q for w in ["sold", "sell", "sales", "out today"]):
            intent = "SALES_TODAY"
        elif any(w in q for w in ["received", "came in", "incoming", "bought", "purchased", "in today"]):
            intent = "RECEIPTS_TODAY"
        elif any(w in q for w in ["review", "pending", "flagged", "approve"]):
            intent = "REVIEW_QUEUE"
        elif any(w in q for w in ["low", "running out", "reorder", "shortage"]):
            intent = "LOW_STOCK"
        elif any(w in q for w in ["history", "transactions", "statements"]):
            intent = "PRODUCT_HISTORY"
        elif any(w in q for w in ["who", "which speaker", "who delivered"]):
            intent = "STATEMENT_SEARCH"
        elif any(w in q for w in ["how much", "how many", "stock", "inventory", "left", "available", "remaining"]):
            intent = "CURRENT_STOCK"
        else:
            intent = "CURRENT_STOCK"

        product_name = None
        patterns = [
            r"(?:how much|how many|what is the stock of|check stock of|quantity of|stock of)\s+(?:bags of\s+|kg of\s+|packets of\s+)?([a-zA-Z0-9_\-]+)",
            r"([a-zA-Z0-9_\-]+)\s+(?:stock|left|remaining|available|inventory)",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                candidate = m.group(1).strip()
                stopwords = {"much", "many", "stock", "left", "the", "a", "an", "is", "are", "do", "we", "have", "any"}
                if candidate not in stopwords and len(candidate) > 1:
                    product_name = candidate
                    break

        if not product_name:
            for word in query.split():
                cleaned_word = re.sub(r"[^\w]", "", word)
                if len(cleaned_word) > 3 and cleaned_word[0].isupper():
                    product_name = cleaned_word
                    break

        return QueryIntent(intent=intent, product_name=product_name, raw_query=query)

    async def generate_explanation(self, context: dict) -> str:
        """Generate explanation based on actual backend evidence."""
        decision = context.get("decision", "UNKNOWN")
        trust_score = context.get("trust_score", 0)
        plausibility_passed = context.get("plausibility_passed", True)
        contradiction = context.get("contradiction_level", "NO_CONFLICT")
        speaker_status = context.get("speaker_status", "UNKNOWN")

        if decision == "AUTO_CONFIRMED":
            parts = []
            if speaker_status == "IDENTIFIED":
                parts.append("the speaker is enrolled and identified")
            if trust_score >= 0.7:
                parts.append(f"speaker trust score is high ({trust_score:.2f})")
            if plausibility_passed:
                parts.append("the quantity is within normal range for this product")
            if contradiction == "NO_CONFLICT":
                parts.append("no conflicting recent statements were found")
            return "Confirmed automatically because " + ", and ".join(parts) + "." if parts else "Confirmed automatically."

        elif decision == "REQUIRES_REVIEW":
            reasons = []
            if speaker_status != "IDENTIFIED":
                reasons.append("speaker could not be identified with confidence")
            if trust_score < 0.5:
                reasons.append(f"speaker trust score is low ({trust_score:.2f})")
            if not plausibility_passed:
                reasons.append("the quantity is outside the normal range for this product")
            if contradiction in ("POSSIBLE_CONFLICT", "STRONG_CONFLICT"):
                reasons.append("a conflicting recent statement was detected")
            return "Requires owner review because " + ", and ".join(reasons) + "." if reasons else "Requires review."

        elif decision == "REJECTED":
            return (
                "Rejected because the statement could not be verified — "
                "speaker is unknown, quantity is implausible, or a strong contradiction was detected."
            )

        return "Decision made by deterministic backend engine."


# ---------------------------------------------------------------------------
# Groq LLM provider (real, using Groq chat API)
# ---------------------------------------------------------------------------

# System prompt kept compact to minimise token cost.
_GROQ_CLAIM_SYSTEM_PROMPT = """\
You are a structured data extraction engine for a retail inventory system.
Your ONLY job is to extract a structured claim from a spoken voice transcript.

Output MUST be a single valid JSON object with EXACTLY these fields:
  product   - string: the product name (e.g. "rice", "oil")
  quantity  - number: numeric quantity (e.g. 20, 5.5); convert word-numbers (twenty -> 20, five -> 5)
  unit      - string: measurement unit in singular form (e.g. "bag", "box", "kg")
  direction - string: exactly "IN" (received/bought/stocked) or "OUT" (sold/dispatched/given)
  actor     - string or null: speaker/person mentioned, null if not mentioned
  event_time - string or null: time reference if mentioned, null otherwise

RULES:
- Output ONLY the JSON object. No markdown, no explanation, no extra text.
- If direction cannot be determined, default to "IN".
- If unit cannot be determined, use "unit".
- If product cannot be determined, use "unknown".
- If quantity cannot be determined or is zero/negative, use 1.
- Convert ALL English number words to digits: one->1, two->2, five->5, ten->10, twenty->20, forty->40, etc.
- product must be lowercase, single or two words max.
- unit must be singular lowercase (bag not bags, box not boxes, kg not kgs).
"""

_GROQ_QUERY_SYSTEM_PROMPT = """\
You are a query intent classifier for a retail inventory system.
Classify the user's natural language query into one of these intents:
CURRENT_STOCK | SALES_TODAY | RECEIPTS_TODAY | PRODUCT_HISTORY | LOW_STOCK | REVIEW_QUEUE | STATEMENT_SEARCH

Output ONLY a valid JSON object with:
  intent       - string: one of the intents above
  product_name - string or null: product mentioned (lowercase), null if none
  date_filter  - string or null: date/time reference if any, null otherwise

No markdown, no explanation.
"""


class GroqLLMProvider(LLMProvider):
    """
    Real LLM provider using Groq chat completions API.

    Requires:
      GROQ_API_KEY  — Groq API key (never logged or returned in responses)
      GROQ_LLM_MODEL — Groq model name (default: llama-3.3-70b-versatile)

    Used when SAATHI_LLM_PROVIDER=groq.

    The LLM ONLY extracts structured data. It NEVER modifies inventory,
    approves/rejects statements, or makes decisions. All output is validated
    by Pydantic (ExtractedClaim) before use.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        from app.config import get_settings
        settings = get_settings()
        self._model = model or settings.GROQ_LLM_MODEL
        # Key is accessed fresh per call via settings to avoid holding it in memory
        self._settings = settings

    @property
    def provider_name(self) -> str:
        return f"groq-{self._model}"

    def _get_api_key(self) -> Optional[str]:
        """Return the Groq API key. Never log or expose this value."""
        return self._settings.GROQ_API_KEY

    def _redact(self, text: str) -> str:
        """Replace the API key with [REDACTED] in any string (defensive)."""
        key = self._get_api_key()
        if key and key in text:
            text = text.replace(key, "[REDACTED]")
        return text

    async def extract_claim(self, transcript: str) -> Optional[ExtractedClaim]:
        """
        Call Groq chat completions to extract a structured inventory claim.
        Returns None on any failure so the pipeline falls back to REQUIRES_REVIEW.
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.warning("GroqLLMProvider: GROQ_API_KEY is not configured; cannot extract claim.")
            return None

        try:
            import groq as groq_sdk
            client = groq_sdk.AsyncGroq(api_key=api_key)

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GROQ_CLAIM_SYSTEM_PROMPT},
                    {"role": "user", "content": f'Transcript: "{transcript}"'},
                ],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or ""
            raw = raw.strip()

            # Parse JSON — the model is instructed to output only JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt to extract embedded JSON
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    logger.warning("GroqLLMProvider: no valid JSON in response for transcript: %r", transcript[:80])
                    return None
                data = json.loads(m.group())

            # Sanitize product field
            product_raw = str(data.get("product") or "").strip()
            product_clean = re.sub(r"[^\w\s]", "", product_raw).strip().title()
            if not product_clean:
                return None

            # Sanitize unit: singular, lowercase
            unit_raw = str(data.get("unit") or "unit").strip().lower()
            unit_raw = unit_raw.rstrip("s") if unit_raw.endswith("s") and len(unit_raw) > 2 else unit_raw
            # Re-normalize known plural roots that shouldn't be stripped
            unit_norm_map = {
                "ba": "bag", "bag": "bag",
                "bo": "box", "box": "box",
                "kg": "kg", "kilogram": "kg",
                "liter": "liter", "litre": "liter",
                "packet": "packet",
                "piece": "piece",
                "ton": "ton",
                "quintal": "quintal",
                "carton": "carton",
                "unit": "unit",
            }
            unit = unit_norm_map.get(unit_raw, unit_raw)

            # Quantity: the LLM should have converted words to numbers already
            qty_raw = data.get("quantity")
            try:
                quantity = float(qty_raw)
            except (TypeError, ValueError):
                # Try word conversion on the raw transcript as a fallback
                quantity = _extract_quantity(transcript)

            direction = str(data.get("direction") or "IN").upper().strip()
            actor = data.get("actor") or None
            if isinstance(actor, str):
                actor = actor.strip() or None
            event_time = data.get("event_time") or None
            if isinstance(event_time, str):
                event_time = event_time.strip() or None

            return ExtractedClaim(
                product=product_clean,
                quantity=quantity,
                unit=unit,
                direction=direction,
                actor=actor,
                event_time=event_time,
            )

        except Exception as exc:
            safe_msg = self._redact(str(exc))
            logger.error("GroqLLMProvider extract_claim error: %s", safe_msg)
            return None

    async def parse_query_intent(self, query: str) -> QueryIntent:
        """
        Call Groq to classify query intent. Falls back to MockLLMProvider on failure.
        """
        api_key = self._get_api_key()
        if not api_key:
            fallback = MockLLMProvider()
            return await fallback.parse_query_intent(query)

        try:
            import groq as groq_sdk
            client = groq_sdk.AsyncGroq(api_key=api_key)

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GROQ_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": f'Query: "{query}"'},
                ],
                temperature=0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )

            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)

            prod = data.get("product_name")
            if prod:
                prod = re.sub(r"[^\w\s]", "", str(prod)).strip().lower() or None

            return QueryIntent(
                intent=str(data.get("intent") or "CURRENT_STOCK"),
                product_name=prod,
                date_filter=data.get("date_filter") or None,
                raw_query=query,
            )
        except Exception as exc:
            safe_msg = self._redact(str(exc))
            logger.warning("GroqLLMProvider parse_query_intent error (falling back): %s", safe_msg)
            fallback = MockLLMProvider()
            return await fallback.parse_query_intent(query)

    async def generate_explanation(self, context: dict) -> str:
        """Deterministic explanation — does not need the real LLM."""
        return await MockLLMProvider().generate_explanation(context)


# ---------------------------------------------------------------------------
# OpenAI-backed provider (legacy, preserved)
# ---------------------------------------------------------------------------

class RealLLMProvider(LLMProvider):
    """
    Real LLM provider using OpenAI API.
    Requires OPENAI_API_KEY in environment or .env.
    """

    @property
    def provider_name(self) -> str:
        return "gpt-4o"

    async def extract_claim(self, transcript: str) -> Optional[ExtractedClaim]:
        try:
            import openai
            from app.config import get_settings
            settings = get_settings()
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                return None

            client = openai.AsyncOpenAI(api_key=api_key)
            prompt = (
                'Extract inventory claim from this voice transcript.\n'
                'Return ONLY valid JSON with fields: product, quantity (number), unit, '
                'direction (IN or OUT), actor, event_time (null if not mentioned).\n'
                f'Transcript: "{transcript}"\nJSON:'
            )
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            if data.get("product"):
                data["product"] = re.sub(r"[^\w\s]", "", str(data["product"])).strip().title()
            return ExtractedClaim(**data)
        except Exception:
            return None

    async def parse_query_intent(self, query: str) -> QueryIntent:
        try:
            import openai
            from app.config import get_settings
            settings = get_settings()
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                return await MockLLMProvider().parse_query_intent(query)

            client = openai.AsyncOpenAI(api_key=api_key)
            intents = "CURRENT_STOCK|SALES_TODAY|RECEIPTS_TODAY|PRODUCT_HISTORY|LOW_STOCK|REVIEW_QUEUE|STATEMENT_SEARCH"
            prompt = (
                f'Classify this inventory query intent.\nIntents: {intents}\n'
                'Return JSON: {"intent": "...", "product_name": null_or_string, "date_filter": null_or_string}\n'
                f'Query: "{query}"\nJSON:'
            )
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON in response")
            data = json.loads(match.group())
            prod = data.get("product_name")
            if prod:
                prod = re.sub(r"[^\w\s]", "", str(prod)).strip()
            return QueryIntent(intent=data.get("intent", "CURRENT_STOCK"), product_name=prod, raw_query=query)
        except Exception:
            return await MockLLMProvider().parse_query_intent(query)

    async def generate_explanation(self, context: dict) -> str:
        return await MockLLMProvider().generate_explanation(context)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    from app.config import get_settings
    settings = get_settings()
    provider = settings.SAATHI_LLM_PROVIDER.lower().strip()
    if provider == "groq":
        return GroqLLMProvider()
    if provider == "openai":
        return RealLLMProvider()
    return MockLLMProvider()
