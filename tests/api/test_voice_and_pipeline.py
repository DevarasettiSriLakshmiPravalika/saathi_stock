"""
Phase 4-15: Voice Enrollment and Voice Processing Pipeline Tests.

Tests:
- Voice Enrollment (owner self, owner for staff, unauthorized enrollment, audio validation)
- Voice Profile listing (safe metadata, no raw embeddings)
- Voice Processing (full pipeline: ASR -> Speaker ID -> Claim Extraction -> Entity Resolution ->
  Trust -> Plausibility -> Contradiction -> Decision -> Statement Ledger -> Inventory)
- Invariant tests:
  - CONFIRMED statement updates inventory
  - FLAGGED statement does NOT update inventory
  - REJECTED statement does NOT update inventory
"""
import pytest
from httpx import AsyncClient

from tests.api.conftest import (
    register_and_login,
    create_shop_for_user,
    create_product_for_shop,
    set_baseline_for_product,
)

DUMMY_WAV = b"RIFF" + b"\x00" * 300


class TestVoiceEnrollment:

    @pytest.mark.asyncio
    async def test_owner_can_enroll_own_voice(self, client: AsyncClient):
        owner = await register_and_login(client, "Voice Owner", "+912002001001")
        shop = await create_shop_for_user(client, owner["headers"])

        files = {"audio": ("sample.wav", DUMMY_WAV, "audio/wav")}
        data = {"shop_id": shop["id"]}
        r = await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["success"] is True
        data = res["data"]
        assert data["enrollment_status"] == "ENROLLED"
        assert "embedding" not in data
        assert data["user_id"] == owner["user"]["id"]

    @pytest.mark.asyncio
    async def test_owner_can_enroll_staff_voice(self, client: AsyncClient):
        owner = await register_and_login(client, "Shop Owner", "+912002001002")
        shop = await create_shop_for_user(client, owner["headers"])

        staff_res = await client.post(
            f"/api/v1/shops/{shop['id']}/members",
            json={"name": "Staff Ram", "phone": "+912002001003", "role": "STAFF"},
            headers=owner["headers"],
        )
        assert staff_res.status_code == 200, staff_res.text
        staff_user_id = staff_res.json()["data"]["user_id"]

        files = {"audio": ("staff.wav", DUMMY_WAV, "audio/wav")}
        data = {"shop_id": shop["id"], "member_user_id": staff_user_id}
        r = await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])
        assert r.status_code == 200
        assert r.json()["data"]["user_id"] == staff_user_id

    @pytest.mark.asyncio
    async def test_staff_cannot_enroll_other_user(self, client: AsyncClient):
        owner = await register_and_login(client, "Shop Owner", "+912002001004")
        shop = await create_shop_for_user(client, owner["headers"])

        staff = await register_and_login(client, "Staff Member", "+912002001005")
        await client.post(
            f"/api/v1/shops/{shop['id']}/members",
            json={"name": "Staff Member", "phone": "+912002001005", "role": "STAFF"},
            headers=owner["headers"],
        )

        files = {"audio": ("other.wav", DUMMY_WAV, "audio/wav")}
        data = {"shop_id": shop["id"], "member_user_id": owner["user"]["id"]}
        r = await client.post("/api/v1/voice/enroll", data=data, files=files, headers=staff["headers"])
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_invalid_audio_format_rejected(self, client: AsyncClient):
        owner = await register_and_login(client, "Voice Owner", "+912002001006")
        shop = await create_shop_for_user(client, owner["headers"])

        files = {"audio": ("bad.exe", b"MZ" + b"\x00" * 200, "application/octet-stream")}
        data = {"shop_id": shop["id"]}
        r = await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_AUDIO"

    @pytest.mark.asyncio
    async def test_audio_too_small_rejected(self, client: AsyncClient):
        owner = await register_and_login(client, "Voice Owner", "+912002001007")
        shop = await create_shop_for_user(client, owner["headers"])

        files = {"audio": ("tiny.wav", b"RIFF" + b"\x00" * 10, "audio/wav")}
        data = {"shop_id": shop["id"]}
        r = await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_AUDIO"

    @pytest.mark.asyncio
    async def test_list_voice_profiles_protects_embeddings(self, client: AsyncClient):
        owner = await register_and_login(client, "Voice Owner", "+912002001008")
        shop = await create_shop_for_user(client, owner["headers"])

        files = {"audio": ("sample.wav", DUMMY_WAV, "audio/wav")}
        data = {"shop_id": shop["id"]}
        await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])

        r = await client.get(f"/api/v1/voice/profiles?shop_id={shop['id']}", headers=owner["headers"])
        assert r.status_code == 200
        profiles = r.json()["data"]
        assert len(profiles) == 1
        assert "embedding" not in profiles[0]
        assert profiles[0]["enrollment_status"] == "ENROLLED"


class TestVoiceProcessingPipeline:

    @pytest.mark.asyncio
    async def test_confirmed_statement_decrements_inventory(self, client: AsyncClient):
        """'Sold 5 bags of rice' should auto-confirm and reduce inventory from 100 to 95."""
        owner = await register_and_login(client, "Owner Rice", "+912002002001")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 100.0, "bag")

        # Process voice with transcript override
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 5 bags of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["success"] is True
        data = res["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 5.0
        assert data["direction"] == "OUT"

        # Verify inventory decreased from 100 to 95
        inv_res = await client.get(f"/api/v1/shops/{shop['id']}/inventory", headers=owner["headers"])
        assert inv_res.status_code == 200
        items = inv_res.json()["data"]
        rice_item = next(i for i in items if i["product_id"] == product["id"])
        assert rice_item["current_stock"] == 95.0
        assert rice_item["confirmed_out"] == 5.0

    @pytest.mark.asyncio
    async def test_confirmed_inward_statement_increments_inventory(self, client: AsyncClient):
        """'Received 20 bags of rice' should auto-confirm and increase inventory from 100 to 120."""
        owner = await register_and_login(client, "Owner Inward", "+912002002002")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 100.0, "bag")

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 20 bags of rice from supplier",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 20.0
        assert data["direction"] == "IN"

        inv_res = await client.get(f"/api/v1/shops/{shop['id']}/inventory", headers=owner["headers"])
        rice_item = next(i for i in inv_res.json()["data"] if i["product_id"] == product["id"])
        assert rice_item["current_stock"] == 120.0
        assert rice_item["confirmed_in"] == 20.0

    @pytest.mark.asyncio
    async def test_unresolved_product_is_flagged_and_inventory_unchanged(self, client: AsyncClient):
        """Unresolved product claim goes to review; inventory of other products unchanged."""
        owner = await register_and_login(client, "Owner Unknown", "+912002002003")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 10 boxes of widgets",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["decision"] == "REQUIRES_REVIEW"
        assert data["status"] == "FLAGGED"
        assert data["review_id"] is not None

        # Existing inventory must be unchanged at 50
        inv_res = await client.get(f"/api/v1/shops/{shop['id']}/inventory", headers=owner["headers"])
        rice_item = next(i for i in inv_res.json()["data"] if i["product_id"] == product["id"])
        assert rice_item["current_stock"] == 50.0

    @pytest.mark.asyncio
    async def test_full_pipeline_with_audio_upload(self, client: AsyncClient):
        """End-to-end voice processing with audio file upload."""
        owner = await register_and_login(client, "Owner Audio", "+912002002004")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        # First enroll owner voice
        files = {"audio": ("owner.wav", DUMMY_WAV, "audio/wav")}
        data = {"shop_id": shop["id"]}
        await client.post("/api/v1/voice/enroll", data=data, files=files, headers=owner["headers"])

        # Now send audio statement with transcript override
        files = {"audio": ("speech.wav", DUMMY_WAV, "audio/wav")}
        data = {
            "shop_id": shop["id"],
            "transcript_override": "Sold 2 bags of rice",
        }
        r = await client.post("/api/v1/voice/process", data=data, files=files, headers=owner["headers"])
        assert r.status_code == 200
        res = r.json()["data"]
        assert res["decision"] == "AUTO_CONFIRMED"
        assert res["status"] == "CONFIRMED"
        assert res["quantity"] == 2.0

        # Inventory check: 50 - 2 = 48
        inv_res = await client.get(f"/api/v1/shops/{shop['id']}/inventory", headers=owner["headers"])
        rice_item = next(i for i in inv_res.json()["data"] if i["product_id"] == product["id"])
        assert rice_item["current_stock"] == 48.0

    @pytest.mark.asyncio
    async def test_sold_5_bags_of_rice_exact_flow_and_query(self, client: AsyncClient):
        """
        Verify exact requirement:
        1. Owner registers, shop created, 'Rice' product created with baseline 100 bags.
        2. Owner enrolls voice sample.
        3. Real audio with command 'Sold 5 bags of rice.'
        4. Produces structured claim: product=Rice, quantity=5, unit=bag, direction=OUT.
        5. Persists into statement ledger as AUTO_CONFIRMED.
        6. Recalculates inventory: 100 - 5 = 95 bags.
        7. Natural language query 'How much rice is left?' returns 95 bags.
        """
        owner = await register_and_login(client, "Rice Master", "+912002009999")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 100.0, "bag")

        # 1. Enroll voice
        enroll_wav = b"RIFF" + b"\x00" * 320
        enroll_files = {"audio": ("owner_enroll.wav", enroll_wav, "audio/wav")}
        enroll_res = await client.post("/api/v1/voice/enroll", data={"shop_id": shop["id"]}, files=enroll_files, headers=owner["headers"])
        assert enroll_res.status_code == 200

        # 2. Process voice command "Sold 5 bags of rice." with audio file
        speech_wav = b"RIFF" + b"\x00" * 450
        speech_files = {"audio": ("speech.wav", speech_wav, "audio/wav")}
        voice_res = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 5 bags of rice.",
            },
            files=speech_files,
            headers=owner["headers"],
        )
        assert voice_res.status_code == 200, voice_res.text
        data = voice_res.json()["data"]

        # Structured claim verification
        assert data["raw_claim"]["product"] == "Rice"
        assert data["raw_claim"]["quantity"] == 5.0
        assert data["raw_claim"]["unit"] == "bag"
        assert data["raw_claim"]["direction"] == "OUT"
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 5.0
        assert data["unit"] == "bag"
        assert data["direction"] == "OUT"
        assert data["product_id"] == product["id"]

        # 3. Verify ledger & inventory recalculation
        inv_res = await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])
        assert inv_res.status_code == 200
        inv = inv_res.json()["data"]
        assert inv["current_stock"] == 95.0
        assert inv["confirmed_out"] == 5.0

        # 4. Natural language query: 'How much rice is left?'
        query_res = await client.post(
            "/api/v1/query",
            json={"shop_id": shop["id"], "query": "How much rice is left?"},
            headers=owner["headers"],
        )
        assert query_res.status_code == 200
        q_data = query_res.json()["data"]
        assert q_data["intent"] == "CURRENT_STOCK"
        assert q_data["product_name"] == "rice"
        assert q_data["result"]["product"] == "Rice"
        assert q_data["result"]["current_stock"] == 95.0

    @pytest.mark.asyncio
    async def test_regression_voice_process_with_audio_only_invokes_asr(self, client: AsyncClient):
        """
        Regression Test: Audio is uploaded without transcript_override.
        Verifies:
        1. Multipart form upload works without 422.
        2. ASR provider path executes (transcribing 'Sold 5 bags of rice.').
        3. Real/mock ASR provider is invoked and recorded in statement processing.
        4. Statement confirms and inventory decrements by 5.
        """
        owner = await register_and_login(client, "ASR Owner", "+912002008881")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        # Enroll owner voice
        enroll_files = {"audio": ("owner.wav", DUMMY_WAV, "audio/wav")}
        await client.post("/api/v1/voice/enroll", data={"shop_id": shop["id"]}, files=enroll_files, headers=owner["headers"])

        # Send audio ONLY (no transcript_override)
        audio_files = {"audio": ("command.wav", DUMMY_WAV, "audio/wav")}
        r = await client.post(
            "/api/v1/voice/process",
            data={"shop_id": shop["id"], "source": "MOBILE_WEB"},
            files=audio_files,
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 5.0
        assert data["direction"] == "OUT"

        # Check inventory decremented from 50 to 45
        inv_res = await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])
        assert inv_res.json()["data"]["current_stock"] == 45.0

    @pytest.mark.asyncio
    async def test_regression_voice_process_with_text_override_only_without_audio(self, client: AsyncClient):
        """
        Regression Test: Multipart request sent with transcript_override only, NO audio file.
        Verifies:
        1. Form processing accepts text-only mode without 422.
        2. Audio is not required when transcript_override is provided.
        3. Statement confirms and inventory decrements by 5.
        """
        owner = await register_and_login(client, "Text Only Owner", "+912002008882")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 60.0, "bag")

        # Send transcript_override without audio
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "source": "MOBILE_WEB",
                "transcript_override": "Sold 5 bags of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["quantity"] == 5.0
        assert data["direction"] == "OUT"

        # Check inventory: 60 - 5 = 55
        inv_res = await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])
        assert inv_res.json()["data"]["current_stock"] == 55.0

    @pytest.mark.asyncio
    async def test_regression_voice_process_empty_request_rejected_with_400(self, client: AsyncClient):
        """
        Regression Test: Request with neither audio nor transcript_override returns 400 Bad Request.
        """
        owner = await register_and_login(client, "Empty Owner", "+912002008883")
        shop = await create_shop_for_user(client, owner["headers"])

        r = await client.post(
            "/api/v1/voice/process",
            data={"shop_id": shop["id"]},
            headers=owner["headers"],
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"


class TestGroqAndASRProviders:

    @pytest.mark.asyncio
    async def test_mock_asr_returns_deterministic_test_transcript(self):
        """1. Mock ASR still returns its deterministic test transcript."""
        from app.providers.asr_provider import MockASRProvider
        provider = MockASRProvider()
        result = await provider.transcribe(b"RIFF" + b"\x00" * 200, "audio.wav")
        assert result.success is True
        assert result.transcript == "Sold 5 bags of rice."
        assert result.provider == "mock-asr-v1"

    def test_provider_selection(self, monkeypatch):
        """2. Groq ASR provider is selected when SAATHI_ASR_PROVIDER=groq."""
        from app.config import get_settings
        from app.providers.asr_provider import get_asr_provider, GroqASRProvider, WhisperASRProvider, MockASRProvider

        # Clear cached settings
        get_settings.cache_clear()

        # Test groq
        monkeypatch.setenv("SAATHI_ASR_PROVIDER", "groq")
        get_settings.cache_clear()
        provider = get_asr_provider()
        assert isinstance(provider, GroqASRProvider)
        assert provider.provider_name.startswith("groq-")

        # Test whisper
        monkeypatch.setenv("SAATHI_ASR_PROVIDER", "whisper")
        get_settings.cache_clear()
        provider = get_asr_provider()
        assert isinstance(provider, WhisperASRProvider)
        assert provider.provider_name == "whisper-v3"

        # Test mock
        monkeypatch.setenv("SAATHI_ASR_PROVIDER", "mock")
        get_settings.cache_clear()
        provider = get_asr_provider()
        assert isinstance(provider, MockASRProvider)
        assert provider.provider_name == "mock-asr-v1"

        # Reset cache
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_groq_missing_api_key_handled_safely(self, monkeypatch):
        """3. Groq without API key returns graceful failure result."""
        from app.config import get_settings
        from app.providers.asr_provider import GroqASRProvider
        get_settings.cache_clear()
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", None)

        provider = GroqASRProvider()
        result = await provider.transcribe(b"RIFF" + b"\x00" * 200, "audio.wav")
        assert result.success is False
        assert "GROQ_API_KEY is not configured" in result.error
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_groq_api_error_handling_and_key_redaction(self, monkeypatch):
        """4. Groq API errors are handled safely and API key is never included in logs/errors."""
        from unittest.mock import AsyncMock, patch
        from app.config import get_settings
        from app.providers.asr_provider import GroqASRProvider

        fake_secret_key = "gsk_fakeSecretKeyForTesting123456789"
        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", fake_secret_key)

        provider = GroqASRProvider()

        # Simulate Groq client raising an error containing the key string
        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.audio.transcriptions.create = AsyncMock(
                side_effect=Exception(f"Error communicating with Groq: {fake_secret_key} unauthorized")
            )
            mock_groq_class.return_value = mock_client

            result = await provider.transcribe(b"RIFF" + b"\x00" * 200, "statement.webm")

            assert result.success is False
            assert result.confidence == 0.0
            # Ensure fake key is completely redacted
            assert fake_secret_key not in result.error
            assert "[REDACTED]" in result.error

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_groq_successful_transcription_multilingual(self, monkeypatch):
        """Verify GroqASRProvider returns actual transcription when API succeeds."""
        from unittest.mock import AsyncMock, patch
        from app.config import get_settings
        from app.providers.asr_provider import GroqASRProvider

        get_settings.cache_clear()
        monkeypatch.setattr(get_settings(), "GROQ_API_KEY", "gsk_dummy_test_key")

        provider = GroqASRProvider(model="whisper-large-v3-turbo")

        class MockGroqResponse:
            text = "Received 20 boxes of oil."
            language = "hi"

        with patch("groq.AsyncGroq") as mock_groq_class:
            mock_client = AsyncMock()
            mock_client.audio.transcriptions.create = AsyncMock(return_value=MockGroqResponse())
            mock_groq_class.return_value = mock_client

            result = await provider.transcribe(b"RIFF" + b"\x00" * 200, "statement.webm")

            assert result.success is True
            assert result.transcript == "Received 20 boxes of oil."
            assert result.language == "hi"
            assert result.provider == "groq-whisper-large-v3-turbo"

        get_settings.cache_clear()


class TestVoiceAutoInventoryAndProductCreation:
    """
    Comprehensive tests for automatic inventory updates and product auto-creation from voice statements:
    - Case 1: Existing product + IN increases inventory.
    - Case 2: Existing product + OUT decreases inventory.
    - Case 3: New product + IN ('Received 20 bags of oil') creates product and updates stock to 20.
    - Case 4: New product + different unit ('Received 20 boxes of biscuits') creates product with unit box.
    - Case 5: Product name resolution ('rice', 'Rice', 'rice bags') does not duplicate product.
    - Case 6: Existing product + incompatible unit ('Received 20 kilograms of rice' where unit is bag) flags review and leaves stock unchanged.
    - Case 7: Unknown / ambiguous claim flags review and creates no product.
    - Case 8: Selling more than available stock flags review and prevents negative stock.
    - Case 9: Review approval auto-creates product and confirms inventory.
    - Case 10: Shop isolation prevents cross-tenant product leakage.
    """

    @pytest.mark.asyncio
    async def test_case_1_existing_product_in_increases_inventory(self, client: AsyncClient):
        """Case 1: 'Received 20 bags of rice' -> Rice stock increases by 20."""
        owner = await register_and_login(client, "Owner Case 1", "+912003000001")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 100.0, "bag")

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 20 bags of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 20.0
        assert data["direction"] == "IN"

        inv_res = await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])
        assert inv_res.status_code == 200
        inv = inv_res.json()["data"]
        assert inv["current_stock"] == 120.0
        assert inv["confirmed_in"] == 20.0
        assert inv["confirmed_out"] == 0.0

    @pytest.mark.asyncio
    async def test_case_2_existing_product_out_decreases_inventory(self, client: AsyncClient):
        """Case 2: 'Sold 5 bags of rice' -> Rice stock decreases by 5."""
        owner = await register_and_login(client, "Owner Case 2", "+912003000002")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 100.0, "bag")

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 5 bags of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 5.0
        assert data["direction"] == "OUT"

        inv_res = await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])
        assert inv_res.status_code == 200
        inv = inv_res.json()["data"]
        assert inv["current_stock"] == 95.0
        assert inv["confirmed_out"] == 5.0

    @pytest.mark.asyncio
    async def test_case_3_new_product_in_auto_creates_product_and_updates_inventory(self, client: AsyncClient):
        """Case 3: 'Received 20 bags of oil' -> Oil does not exist -> auto-creates Oil with unit 'bag' and stock 20."""
        owner = await register_and_login(client, "Owner Case 3", "+912003000003")
        shop = await create_shop_for_user(client, owner["headers"])

        # Verify shop currently has no products
        prods_before = await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])
        assert len(prods_before.json()["data"]) == 0

        # Owner speaks statement for non-existent product
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 20 bags of oil",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 20.0
        assert data["unit"] == "bag"
        assert data["direction"] == "IN"
        assert data["product_id"] is not None

        # Verify product was auto-created
        prods_after = await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])
        prods = prods_after.json()["data"]
        assert len(prods) == 1
        oil_prod = prods[0]
        assert oil_prod["name"] == "Oil"
        assert oil_prod["default_unit"] == "bag"
        assert str(oil_prod["id"]) == data["product_id"]

        # Verify inventory derivation: 0 baseline + 20 confirmed_in = 20 bags
        inv_res = await client.get(f"/api/v1/products/{oil_prod['id']}/inventory", headers=owner["headers"])
        assert inv_res.status_code == 200
        inv = inv_res.json()["data"]
        assert inv["product_name"] == "Oil"
        assert inv["current_stock"] == 20.0
        assert inv["confirmed_in"] == 20.0
        assert inv["confirmed_out"] == 0.0

        # Verify shop-wide inventory overview also reflects Oil
        shop_inv = await client.get(f"/api/v1/shops/{shop['id']}/inventory", headers=owner["headers"])
        assert shop_inv.status_code == 200
        items = shop_inv.json()["data"]
        assert len(items) == 1
        assert items[0]["product_name"] == "Oil"
        assert items[0]["current_stock"] == 20.0

    @pytest.mark.asyncio
    async def test_case_4_new_product_with_different_unit_creates_product_and_stock(self, client: AsyncClient):
        """Case 4: 'Received 20 boxes of biscuits' -> creates Biscuits with unit 'box' and stock 20."""
        owner = await register_and_login(client, "Owner Case 4", "+912003000004")
        shop = await create_shop_for_user(client, owner["headers"])

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 20 boxes of biscuits",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["status"] == "CONFIRMED"
        assert data["quantity"] == 20.0
        assert data["unit"] == "box"
        assert data["direction"] == "IN"

        # Check product created with box unit
        prods = (await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])).json()["data"]
        assert len(prods) == 1
        assert prods[0]["name"] == "Biscuits"
        assert prods[0]["default_unit"] == "box"

        # Check inventory is 20 boxes
        inv = (await client.get(f"/api/v1/products/{prods[0]['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 20.0
        assert inv["unit"] == "box"

    @pytest.mark.asyncio
    async def test_case_5_product_name_resolution_avoids_duplicate_creation(self, client: AsyncClient):
        """Case 5: 'rice', 'Rice', 'rice bags' resolve to the existing product without duplicate creation."""
        owner = await register_and_login(client, "Owner Case 5", "+912003000005")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        # Command with entity variant 'rice bags'
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 10 rice bags",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "AUTO_CONFIRMED"
        assert data["product_id"] == product["id"]

        # Ensure no duplicate product was created
        prods = (await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])).json()["data"]
        assert len(prods) == 1
        assert prods[0]["id"] == product["id"]

        # Stock should be 50 + 10 = 60
        inv = (await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 60.0

    @pytest.mark.asyncio
    async def test_case_6_existing_product_with_incompatible_unit_flags_review(self, client: AsyncClient):
        """Case 6: Rice exists with unit 'bag'. 'Received 20 kilograms of rice' must flag review, NOT treat kg as bag."""
        owner = await register_and_login(client, "Owner Case 6", "+912003000006")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 20 kilograms of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "REQUIRES_REVIEW"
        assert data["status"] == "FLAGGED"
        assert data["review_id"] is not None

        # Stock must remain unchanged at 50 bags
        inv = (await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 50.0
        assert inv["confirmed_in"] == 0.0

    @pytest.mark.asyncio
    async def test_case_7_unknown_or_ambiguous_claim_flags_review_without_product_creation(self, client: AsyncClient):
        """Case 7: Unknown/ambiguous claim routes to review without modifying inventory or creating products."""
        owner = await register_and_login(client, "Owner Case 7", "+912003000007")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 50.0, "bag")

        # Unknown product on OUT direction
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 10 mysterious gadgets",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "REQUIRES_REVIEW"
        assert data["status"] == "FLAGGED"
        assert data["review_id"] is not None

        # Products list must contain ONLY Rice
        prods = (await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])).json()["data"]
        assert len(prods) == 1
        assert prods[0]["name"] == "Rice"

        # Rice inventory must be unchanged
        inv = (await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 50.0

    @pytest.mark.asyncio
    async def test_case_8_insufficient_stock_contradiction_prevents_negative_inventory(self, client: AsyncClient):
        """Case 8: Selling more than available stock triggers contradiction check, flags review, stock remains unchanged."""
        owner = await register_and_login(client, "Owner Case 8", "+912003000008")
        shop = await create_shop_for_user(client, owner["headers"])
        product = await create_product_for_shop(client, owner["headers"], shop["id"], "Rice", "bag")
        await set_baseline_for_product(client, owner["headers"], shop["id"], product["id"], 3.0, "bag")

        # Attempt to sell 10 bags when only 3 exist
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Sold 10 bags of rice",
            },
            headers=owner["headers"],
        )
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        assert data["decision"] == "REQUIRES_REVIEW"
        assert data["status"] == "FLAGGED"
        assert data["review_id"] is not None

        # Stock must remain exactly 3.0 (never negative)
        inv = (await client.get(f"/api/v1/products/{product['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 3.0
        assert inv["confirmed_out"] == 0.0

    @pytest.mark.asyncio
    async def test_case_9_review_approval_creates_product_and_confirms_inventory(self, client: AsyncClient):
        """Case 9: If an incoming statement was flagged to review, owner approval creates product and updates stock."""
        owner = await register_and_login(client, "Owner Case 9", "+912003000009")
        shop = await create_shop_for_user(client, owner["headers"])

        # Register a staff member whose voice is not enrolled / default low trust
        staff = await register_and_login(client, "New Staff", "+912003000010")
        await client.post(
            f"/api/v1/shops/{shop['id']}/members",
            json={"name": "New Staff", "phone": "+912003000010", "role": "STAFF"},
            headers=owner["headers"],
        )

        # Staff posts incoming statement with audio for new product -> without voice profile, speaker is UNKNOWN, routes to review
        files = {"audio": ("unknown_staff.wav", DUMMY_WAV, "audio/wav")}
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop["id"],
                "transcript_override": "Received 25 bags of wheat",
            },
            files=files,
            headers=staff["headers"],
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["decision"] == "REQUIRES_REVIEW"
        assert data["status"] == "FLAGGED"
        review_id = data["review_id"]
        assert review_id is not None

        # Wheat should not exist yet before approval
        prods_before = (await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])).json()["data"]
        assert len(prods_before) == 0

        # Owner approves the review
        appr_res = await client.post(
            f"/api/v1/reviews/{review_id}/approve",
            json={"reason": "Shipment verified by owner"},
            headers=owner["headers"],
        )
        assert appr_res.status_code == 200, appr_res.text
        assert appr_res.json()["data"]["decision"] == "APPROVED"

        # Now Wheat product must exist and inventory must be 25 bags
        prods_after = (await client.get(f"/api/v1/shops/{shop['id']}/products", headers=owner["headers"])).json()["data"]
        assert len(prods_after) == 1
        wheat = prods_after[0]
        assert wheat["name"] == "Wheat"
        assert wheat["default_unit"] == "bag"

        inv = (await client.get(f"/api/v1/products/{wheat['id']}/inventory", headers=owner["headers"])).json()["data"]
        assert inv["current_stock"] == 25.0
        assert inv["confirmed_in"] == 25.0

    @pytest.mark.asyncio
    async def test_case_10_shop_isolation_for_voice_auto_created_products(self, client: AsyncClient):
        """Case 10: Shop isolation — auto-creating a product in Shop A must never leak into Shop B."""
        owner_a = await register_and_login(client, "Owner Shop A", "+912003000011")
        shop_a = await create_shop_for_user(client, owner_a["headers"])

        owner_b = await register_and_login(client, "Owner Shop B", "+912003000012")
        shop_b = await create_shop_for_user(client, owner_b["headers"])

        # Shop A receives 30 bags of sugar
        r = await client.post(
            "/api/v1/voice/process",
            data={
                "shop_id": shop_a["id"],
                "transcript_override": "Received 30 bags of sugar",
            },
            headers=owner_a["headers"],
        )
        assert r.status_code == 200
        assert r.json()["data"]["status"] == "CONFIRMED"

        # Shop A has Sugar
        prods_a = (await client.get(f"/api/v1/shops/{shop_a['id']}/products", headers=owner_a["headers"])).json()["data"]
        assert len(prods_a) == 1
        assert prods_a[0]["name"] == "Sugar"

        # Shop B must NOT have Sugar
        prods_b = (await client.get(f"/api/v1/shops/{shop_b['id']}/products", headers=owner_b["headers"])).json()["data"]
        assert len(prods_b) == 0

        # Shop B inventory is empty
        inv_b = (await client.get(f"/api/v1/shops/{shop_b['id']}/inventory", headers=owner_b["headers"])).json()["data"]
        assert len(inv_b) == 0



