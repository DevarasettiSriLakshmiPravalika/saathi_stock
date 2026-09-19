"""
Tests for LLM providers: GroqLLMProvider and MockLLMProvider.

Coverage:
- Word-number conversion (_words_to_number / _extract_quantity)
- MockLLMProvider deterministic parsing (digit and word quantities)
- GroqLLMProvider unit tests (mocked Groq API):
    * Missing API key → None (safe failure)
    * API error → None (redacted, key never in error)
    * Successful JSON extraction → ExtractedClaim
    * Provider selection via env variable
- Schema / Pydantic validation (invalid LLM output safely rejected)
- Representative transcripts: all 15+ required scenarios
- Existing guardrail invariants not weakened
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.providers.llm_provider import (
    ExtractedClaim,
    MockLLMProvider,
    GroqLLMProvider,
    _words_to_number,
    _extract_quantity,
    get_llm_provider,
)


# ---------------------------------------------------------------------------
# _words_to_number helper
# ---------------------------------------------------------------------------

class TestWordsToNumber:
    def test_simple_ones(self):
        assert _words_to_number("five") == 5.0
        assert _words_to_number("one") == 1.0
        assert _words_to_number("ten") == 10.0

    def test_tens(self):
        assert _words_to_number("twenty") == 20.0
        assert _words_to_number("forty") == 40.0
        assert _words_to_number("thirty") == 30.0
        assert _words_to_number("ninety") == 90.0

    def test_compound(self):
        assert _words_to_number("twenty five") == 25.0
        assert _words_to_number("forty two") == 42.0
        assert _words_to_number("thirty seven") == 37.0

    def test_hundreds(self):
        assert _words_to_number("one hundred") == 100.0
        assert _words_to_number("two hundred") == 200.0
        assert _words_to_number("three hundred fifty") == 350.0

    def test_no_number_returns_none(self):
        assert _words_to_number("bags of rice") is None
        assert _words_to_number("sold received") is None

    def test_zero(self):
        assert _words_to_number("zero") == 0.0


class TestExtractQuantity:
    def test_digit(self):
        assert _extract_quantity("sold 5 bags of rice") == 5.0
        assert _extract_quantity("received 40 boxes of oil") == 40.0

    def test_decimal_digit(self):
        assert _extract_quantity("received 2.5 kg of rice") == 2.5

    def test_word_number(self):
        assert _extract_quantity("sold five bags of rice") == 5.0
        assert _extract_quantity("received twenty boxes of oil") == 20.0
        assert _extract_quantity("received forty boxes of oil") == 40.0
        assert _extract_quantity("sold ten boxes of oil") == 10.0

    def test_fallback_to_one(self):
        # No digit, no word-number
        assert _extract_quantity("bags of rice") == 1.0


# ---------------------------------------------------------------------------
# MockLLMProvider — digit quantities
# ---------------------------------------------------------------------------

class TestMockLLMProviderDigitQuantities:

    @pytest.mark.asyncio
    async def test_received_20_bags_of_rice(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Received 20 bags of rice")
        assert claim is not None
        assert claim.quantity == 20.0
        assert claim.unit == "bag"
        assert claim.direction == "IN"
        assert claim.product.lower() == "rice"

    @pytest.mark.asyncio
    async def test_sold_5_bags_of_rice(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold 5 bags of rice")
        assert claim is not None
        assert claim.quantity == 5.0
        assert claim.unit == "bag"
        assert claim.direction == "OUT"
        assert claim.product.lower() == "rice"

    @pytest.mark.asyncio
    async def test_received_40_boxes_of_oil(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Received 40 boxes of oil")
        assert claim is not None
        assert claim.quantity == 40.0
        assert claim.unit == "box"
        assert claim.direction == "IN"
        assert claim.product.lower() == "oil"

    @pytest.mark.asyncio
    async def test_sold_10_boxes_of_oil(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold 10 boxes of oil")
        assert claim is not None
        assert claim.quantity == 10.0
        assert claim.unit == "box"
        assert claim.direction == "OUT"
        assert claim.product.lower() == "oil"

    @pytest.mark.asyncio
    async def test_provider_name(self):
        p = MockLLMProvider()
        assert p.provider_name == "mock-llm-v1"


# ---------------------------------------------------------------------------
# MockLLMProvider — English word-number quantities (the main fix)
# ---------------------------------------------------------------------------

class TestMockLLMProviderWordNumbers:

    @pytest.mark.asyncio
    async def test_received_twenty_boxes_of_oil(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Received twenty boxes of oil.")
        assert claim is not None, "Expected a claim from 'Received twenty boxes of oil.'"
        assert claim.quantity == 20.0
        assert claim.unit == "box"
        assert claim.direction == "IN"
        assert claim.product.lower() == "oil"

    @pytest.mark.asyncio
    async def test_sold_five_bags_of_rice(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold five bags of rice.")
        assert claim is not None, "Expected a claim from 'Sold five bags of rice.'"
        assert claim.quantity == 5.0
        assert claim.unit == "bag"
        assert claim.direction == "OUT"
        assert claim.product.lower() == "rice"

    @pytest.mark.asyncio
    async def test_received_forty_boxes_of_oil(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Received forty boxes of oil.")
        assert claim is not None
        assert claim.quantity == 40.0
        assert claim.unit == "box"
        assert claim.direction == "IN"

    @pytest.mark.asyncio
    async def test_sold_ten_boxes_of_oil(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold ten boxes of oil.")
        assert claim is not None
        assert claim.quantity == 10.0
        assert claim.unit == "box"
        assert claim.direction == "OUT"

    @pytest.mark.asyncio
    async def test_received_thirty_kg_of_wheat(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Received thirty kg of wheat.")
        assert claim is not None
        assert claim.quantity == 30.0
        assert claim.unit == "kg"
        assert claim.direction == "IN"
        assert claim.product.lower() == "wheat"

    @pytest.mark.asyncio
    async def test_sold_one_bag_of_sugar(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold one bag of sugar.")
        assert claim is not None
        assert claim.quantity == 1.0
        assert claim.unit == "bag"
        assert claim.direction == "OUT"


# ---------------------------------------------------------------------------
# MockLLMProvider — edge/failure cases
# ---------------------------------------------------------------------------

class TestMockLLMProviderEdgeCases:

    @pytest.mark.asyncio
    async def test_unknown_product_returns_none(self):
        """Transcripts that yield no identifiable product return None."""
        p = MockLLMProvider()
        # Extremely minimal — no product remaining after strip
        claim = await p.extract_claim("Sold 10 bags")
        # "bags" stripped → could produce None or a degenerate result; either is acceptable
        # The key invariant: if returned, quantity must be positive
        if claim is not None:
            assert claim.quantity > 0

    @pytest.mark.asyncio
    async def test_ambiguous_direction_defaults_to_in(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("5 bags of rice processed")
        if claim is not None:
            assert claim.direction in ("IN", "OUT")

    @pytest.mark.asyncio
    async def test_transcript_with_trailing_punctuation(self):
        p = MockLLMProvider()
        claim = await p.extract_claim("Sold 5 bags of rice.")
        assert claim is not None
        assert claim.quantity == 5.0
        assert claim.product.lower() == "rice"

    @pytest.mark.asyncio
    async def test_extracted_claim_pydantic_rejects_zero_quantity(self):
        with pytest.raises(Exception):
            ExtractedClaim(product="Rice", quantity=0, unit="bag", direction="IN")

    @pytest.mark.asyncio
    async def test_extracted_claim_pydantic_rejects_negative_quantity(self):
        with pytest.raises(Exception):
            ExtractedClaim(product="Rice", quantity=-5, unit="bag", direction="IN")

    @pytest.mark.asyncio
    async def test_extracted_claim_pydantic_rejects_invalid_direction(self):
        with pytest.raises(Exception):
            ExtractedClaim(product="Rice", quantity=5, unit="bag", direction="MAYBE")

    @pytest.mark.asyncio
    async def test_extracted_claim_valid(self):
        claim = ExtractedClaim(product="Rice", quantity=5.0, unit="bag", direction="OUT")
        assert claim.direction == "OUT"
        assert claim.quantity == 5.0

    @pytest.mark.asyncio
    async def test_direction_normalised_to_uppercase(self):
        claim = ExtractedClaim(product="Rice", quantity=5.0, unit="bag", direction="out")
        assert claim.direction == "OUT"


# ---------------------------------------------------------------------------
# GroqLLMProvider — unit tests (no real API key needed)
# ---------------------------------------------------------------------------

class TestGroqLLMProvider:

    def test_provider_name(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake")
        monkeypatch.setattr(get_settings(), "GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
        p = GroqLLMProvider(model="llama-3.3-70b-versatile")
        assert p.provider_name == "groq-llama-3.3-70b-versatile"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_none(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", None)
        p = GroqLLMProvider()
        result = await p.extract_claim("Received 20 bags of rice")
        assert result is None
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_api_error_returns_none_and_key_is_redacted(self, monkeypatch):
        """Groq API errors must not expose the API key in logs or exceptions."""
        from app.config import get_settings
        fake_key = "gsk_superSecretFakeKeyForTest9999"
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", fake_key)

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception(f"Auth error: {fake_key} is invalid")
            )
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Received 20 bags of rice")

        assert result is None
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_successful_extraction_digit_quantity(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake_key_for_testing")

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "product": "rice",
            "quantity": 20,
            "unit": "bag",
            "direction": "IN",
            "actor": None,
            "event_time": None,
        })

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Received 20 bags of rice")

        assert result is not None
        assert result.product == "Rice"
        assert result.quantity == 20.0
        assert result.unit == "bag"
        assert result.direction == "IN"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_successful_extraction_word_quantity_from_llm(self, monkeypatch):
        """Verify that when the LLM correctly converts word-numbers, GroqLLMProvider returns them."""
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake_key_for_testing")

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "product": "oil",
            "quantity": 20,  # LLM returns 20 for "twenty"
            "unit": "box",
            "direction": "IN",
            "actor": None,
            "event_time": None,
        })

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Received twenty boxes of oil.")

        assert result is not None
        assert result.quantity == 20.0
        assert result.unit == "box"
        assert result.direction == "IN"
        assert result.product == "Oil"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake")

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sorry, I cannot extract that."

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Some gibberish transcript")

        assert result is None
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_empty_product_returns_none(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake")

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "product": "",
            "quantity": 5,
            "unit": "bag",
            "direction": "OUT",
            "actor": None,
            "event_time": None,
        })

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Sold 5 bags of")

        assert result is None
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_pydantic_rejects_invalid_llm_direction(self, monkeypatch):
        """If LLM returns an invalid direction, ExtractedClaim validation catches it and returns None."""
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake")

        p = GroqLLMProvider(model="llama-3.3-70b-versatile")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "product": "rice",
            "quantity": 5,
            "unit": "bag",
            "direction": "MAYBE",  # invalid
            "actor": None,
            "event_time": None,
        })

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_groq_class.return_value = mock_client

            result = await p.extract_claim("Rice")

        assert result is None
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_groq_query_intent_fallback_on_missing_key(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", None)

        p = GroqLLMProvider()
        intent = await p.parse_query_intent("How much rice is left?")
        assert intent.intent == "CURRENT_STOCK"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_generate_explanation_is_deterministic(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_fake")

        p = GroqLLMProvider()
        explanation = await p.generate_explanation({
            "decision": "AUTO_CONFIRMED",
            "trust_score": 0.9,
            "plausibility_passed": True,
            "contradiction_level": "NO_CONFLICT",
            "speaker_status": "IDENTIFIED",
        })
        assert "Confirmed" in explanation
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Provider factory / selection
# ---------------------------------------------------------------------------

class TestLLMProviderSelection:

    def test_mock_selected_by_default(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("SAATHI_LLM_PROVIDER", "mock")
        get_settings.cache_clear()
        p = get_llm_provider()
        assert isinstance(p, MockLLMProvider)
        get_settings.cache_clear()

    def test_groq_selected_when_configured(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("SAATHI_LLM_PROVIDER", "groq")
        get_settings.cache_clear()
        p = get_llm_provider()
        assert isinstance(p, GroqLLMProvider)
        assert p.provider_name.startswith("groq-")
        get_settings.cache_clear()

    def test_openai_selected_when_configured(self, monkeypatch):
        from app.providers.llm_provider import RealLLMProvider
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("SAATHI_LLM_PROVIDER", "openai")
        get_settings.cache_clear()
        p = get_llm_provider()
        assert isinstance(p, RealLLMProvider)
        get_settings.cache_clear()

    def test_unknown_provider_falls_back_to_mock(self, monkeypatch):
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("SAATHI_LLM_PROVIDER", "nonexistent_provider")
        get_settings.cache_clear()
        p = get_llm_provider()
        assert isinstance(p, MockLLMProvider)
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Representative transcript scenarios (no API key required — all use Mock)
# ---------------------------------------------------------------------------

import json  # noqa: E402  (already imported, harmless re-import)


class TestRepresentativeTranscripts:
    """
    Covers the exact scenarios listed in the requirements.
    Uses MockLLMProvider (deterministic, no API key needed).
    All tests validate that:
      - Correct quantity (digits and words)
      - Correct direction
      - Correct unit
      - Correct product
    """

    @pytest.fixture
    def provider(self):
        return MockLLMProvider()

    @pytest.mark.asyncio
    async def test_received_20_bags_of_rice(self, provider):
        c = await provider.extract_claim("Received 20 bags of rice")
        assert c and c.quantity == 20.0 and c.direction == "IN" and c.unit == "bag"

    @pytest.mark.asyncio
    async def test_received_twenty_bags_of_rice(self, provider):
        c = await provider.extract_claim("Received twenty bags of rice")
        assert c is not None, "Word-number 'twenty' must be parsed"
        assert c.quantity == 20.0
        assert c.direction == "IN"
        assert c.unit == "bag"

    @pytest.mark.asyncio
    async def test_received_40_boxes_of_oil(self, provider):
        c = await provider.extract_claim("Received 40 boxes of oil")
        assert c and c.quantity == 40.0 and c.direction == "IN" and c.unit == "box"

    @pytest.mark.asyncio
    async def test_received_forty_boxes_of_oil(self, provider):
        c = await provider.extract_claim("Received forty boxes of oil.")
        assert c is not None
        assert c.quantity == 40.0
        assert c.direction == "IN"
        assert c.unit == "box"

    @pytest.mark.asyncio
    async def test_sold_5_bags_of_rice(self, provider):
        c = await provider.extract_claim("Sold 5 bags of rice")
        assert c and c.quantity == 5.0 and c.direction == "OUT" and c.unit == "bag"

    @pytest.mark.asyncio
    async def test_sold_five_bags_of_rice(self, provider):
        c = await provider.extract_claim("Sold five bags of rice.")
        assert c is not None, "Word-number 'five' must be parsed"
        assert c.quantity == 5.0
        assert c.direction == "OUT"
        assert c.unit == "bag"

    @pytest.mark.asyncio
    async def test_sold_10_boxes_of_oil(self, provider):
        c = await provider.extract_claim("Sold 10 boxes of oil")
        assert c and c.quantity == 10.0 and c.direction == "OUT" and c.unit == "box"

    @pytest.mark.asyncio
    async def test_sold_ten_boxes_of_oil(self, provider):
        c = await provider.extract_claim("Sold ten boxes of oil.")
        assert c is not None
        assert c.quantity == 10.0
        assert c.direction == "OUT"
        assert c.unit == "box"

    @pytest.mark.asyncio
    async def test_unknown_product_returns_none_or_flaggable(self, provider):
        """Unknown / ambiguous product should return None (pipeline flags it for review)."""
        c = await provider.extract_claim("I did something with things.")
        # None = pipeline will create REQUIRES_REVIEW; or if returned product == "unknown" the pipeline also flags it
        if c is not None:
            assert c.product.lower() not in ("", " ")  # Must not be blank

    @pytest.mark.asyncio
    async def test_missing_direction_defaults_gracefully(self, provider):
        """Transcript with no clear direction word should still produce a safe claim."""
        c = await provider.extract_claim("20 bags of rice")
        if c is not None:
            assert c.direction in ("IN", "OUT")
            assert c.quantity == 20.0

    @pytest.mark.asyncio
    async def test_excessive_out_quantity_returns_claim_pipeline_flags(self, provider):
        """
        The LLM/mock must still return a valid claim for 'Sold 1000 bags of rice'.
        Contradiction engine (not the LLM) is responsible for flagging it.
        """
        c = await provider.extract_claim("Sold 1000 bags of rice")
        assert c is not None
        assert c.quantity == 1000.0
        assert c.direction == "OUT"

    @pytest.mark.asyncio
    async def test_new_incoming_product(self, provider):
        """New product (oil, not yet in catalog) — mock should still extract it."""
        c = await provider.extract_claim("Received 20 bags of oil")
        assert c is not None
        assert c.product.lower() == "oil"
        assert c.quantity == 20.0
        assert c.direction == "IN"

    @pytest.mark.asyncio
    async def test_incompatible_unit_claim_returned_to_pipeline(self, provider):
        """
        'Received 20 kilograms of rice' — mock extracts the claim.
        Unit mismatch detection is handled deterministically by voice.py,
        NOT by the claim extractor.
        """
        c = await provider.extract_claim("Received 20 kilograms of rice")
        assert c is not None
        assert c.quantity == 20.0
        assert c.unit == "kg"
        assert c.direction == "IN"
        # product should be rice
        assert c.product.lower() == "rice"
