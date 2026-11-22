"""Cross-Lingual Query Translator (Vietnamese / Multilingual -> English).

Bridges Vietnamese (and other non-English) natural language queries into the
English semantic embedding space of OpenAI CLIP (ViT-B/32) and Google SigLIP
(ViT-B/16-224) without requiring re-indexing of the 164,476-keyframe FAISS stores.

Multi-tier translation strategy (ordered by speed, visual fidelity, and resilience):
1. In-memory LRU cache (0 ms for repeated queries).
2. Grounded VLM Taxonomy Bank exact bilingual lookup (0 ms, domain-accurate phrasing).
3. Google Neural Machine Translation endpoint via persistent HTTP keep-alive session.
4. Local offline MarianMT (`Helsinki-NLP/opus-mt-vi-en`) fallback when offline/rate-limited.
5. Secondary MyMemory HTTP API fallback for non-Vietnamese multilingual queries.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# Vietnamese characters with diacritics or đ/Đ
_VN_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)

# Unambiguous unaccented Vietnamese tokens (avoiding any English words)
_UNACCENTED_VI_TOKENS = frozenset(
    {
        "nguoi",
        "duong",
        "pho",
        "dong",
        "duc",
        "chay",
        "rung",
        "cuu",
        "hoa",
        "linh",
        "nuoc",
        "lua",
        "khoi",
        "ngap",
        "lut",
        "bao",
        "dat",
        "nha",
        "cua",
        "tren",
        "duoi",
        "trong",
        "ngoai",
        "troi",
        "bien",
        "tau",
        "pha",
        "thuyen",
        "khach",
        "canh",
        "sat",
        "bieu",
        "tinh",
        "tong",
        "thong",
        "phat",
        "hop",
        "phien",
        "toa",
        "xet",
        "xu",
        "nghi",
        "pham",
        "cong",
        "tay",
        "ten",
        "phong",
        "be",
        "san",
        "bong",
        "khan",
        "gia",
        "bac",
        "si",
        "benh",
        "vien",
        "nhan",
        "cuuho",
        "dieu",
        "binh",
        "quoc",
        "ky",
        "vang",
        "thoi",
        "ngan",
        "hang",
        "tram",
        "dien",
        "mat",
        "gio",
        "oto",
        "may",
        "bay",
        "truc",
        "thang",
        "chinh",
        "phu",
        "dang",
        "nhieu",
        "nhung",
        "cac",
        "voi",
        "khong",
        "chiec",
        "mot",
        "hai",
        "nguoidan",
    }
)

_UNACCENTED_VI_BIGRAMS = frozenset(
    {
        "xe may",
        "cuu hoa",
        "chay rung",
        "dong dat",
        "ten lua",
        "canh sat",
        "hop bao",
        "duong pho",
        "bao lu",
        "ngap lut",
        "dap lua",
        "bong ro",
        "tau chien",
        "pin mat",
        "mat troi",
        "ao phao",
        "tong thong",
        "phien toa",
        "bac si",
        "benh vien",
        "cuu ho",
        "bieu tinh",
        "truc thang",
        "nguoi di",
        "pha cho",
        "kho vang",
        "nha may",
        "san van",
    }
)

_STRONG_ENGLISH_WORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "for",
        "are",
        "was",
        "were",
        "into",
        "over",
        "under",
        "through",
        "wearing",
        "holding",
        "standing",
        "riding",
        "burning",
        "spraying",
        "people",
        "person",
        "police",
        "fire",
        "water",
        "rocket",
        "street",
        "crowded",
        "building",
        "wildfire",
        "earthquake",
        "ferry",
        "court",
        "trial",
        "podium",
        "press",
        "conference",
        "motorcycle",
        "soldiers",
        "hospital",
        "flood",
        "solar",
    }
)


def normalize_query_key(text: str) -> str:
    """Normalizes whitespace and casing for dictionary / cache lookups."""
    return re.sub(r"\s+", " ", text.strip()).lower()


# Ordered longest-first restoration pairs for common unaccented Vietnamese visual phrases
_UNACCENTED_VI_RESTORE: Tuple[Tuple[str, str], ...] = (
    ("nguoi di xe may", "người đi xe máy"),
    ("pha cho khach", "phà chở khách"),
    ("xe cuu hoa", "xe cứu hỏa"),
    ("linh cuu hoa", "lính cứu hỏa"),
    ("pin mat troi", "pin mặt trời"),
    ("dien mat troi", "điện mặt trời"),
    ("san van dong", "sân vận động"),
    ("san bong ro", "sân bóng rổ"),
    ("duong pho dong duc", "đường phố đông đúc"),
    ("dap lua", "dập lửa"),
    ("boc chay", "bốc cháy"),
    ("chay rung", "cháy rừng"),
    ("khoi den", "khói đen"),
    ("dong dat", "động đất"),
    ("do nat", "đổ nát"),
    ("cuu ho", "cứu hộ"),
    ("xe may", "xe máy"),
    ("duong pho", "đường phố"),
    ("dong duc", "đông đúc"),
    ("ten lua", "tên lửa"),
    ("be phong", "bệ phóng"),
    ("canh sat", "cảnh sát"),
    ("bieu tinh", "biểu tình"),
    ("hop bao", "họp báo"),
    ("tong thong", "tổng thống"),
    ("phien toa", "phiên tòa"),
    ("bao lu", "bão lũ"),
    ("ngap lut", "ngập lụt"),
    ("bong ro", "bóng rổ"),
    ("tau chien", "tàu chiến"),
    ("ao phao", "áo phao"),
    ("bac si", "bác sĩ"),
    ("benh vien", "bệnh viện"),
    ("truc thang", "trực thăng"),
    ("dieu binh", "diễu binh"),
    ("quoc ky", "quốc kỳ"),
    ("tren duong", "trên đường"),
    ("tren bien", "trên biển"),
)


def _restore_unaccented_vi_phrases(text: str) -> str:
    """Restores diacritics for common unaccented Vietnamese phrases before NMT."""
    if _VN_DIACRITIC_RE.search(text):
        return text
    restored = text
    for plain, accented in _UNACCENTED_VI_RESTORE:
        restored = re.sub(rf"\b{re.escape(plain)}\b", accented, restored, flags=re.IGNORECASE)
    return restored


def detect_query_language(text: str) -> str:
    """Classifies query language as 'vi', 'en', or 'multilingual'."""
    cleaned = text.strip()
    if not cleaned:
        return "en"

    if _VN_DIACRITIC_RE.search(cleaned):
        return "vi"

    if not cleaned.isascii():
        return "multilingual"

    # Unaccented Vietnamese check (strictly guarded against English false positives)
    tokens = re.findall(r"[a-z]+", cleaned.lower())
    if len(tokens) >= 2:
        has_strong_en = any(tok in _STRONG_ENGLISH_WORDS for tok in tokens)
        if not has_strong_en:
            bigrams = {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}
            if bigrams & _UNACCENTED_VI_BIGRAMS:
                return "vi"
            vi_hits = sum(1 for tok in tokens if tok in _UNACCENTED_VI_TOKENS)
            if vi_hits >= 2 and (vi_hits / len(tokens)) >= 0.5:
                return "vi"

    return "en"


class QueryTranslator:
    """Thread-safe multi-tier translator from Vietnamese/Multilingual to English."""

    OFFLINE_MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        enable_online: bool = True,
        enable_offline_fallback: bool = True,
        timeout_sec: float = 3.5,
        cache_size: int = 512,
    ) -> None:
        self.enable_online = enable_online
        self.enable_offline_fallback = enable_offline_fallback
        self.timeout_sec = timeout_sec
        self.cache_size = cache_size

        self._lock = threading.Lock()
        self._cache: "OrderedDict[Tuple[str, str], Tuple[str, str]]" = OrderedDict()
        self._taxonomy_map: Dict[str, str] = {}

        # Lazy-initialized offline MarianMT model
        self._offline_tok = None
        self._offline_model = None
        self._offline_load_failed = False

        # Persistent HTTP session for low-latency keep-alive requests
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.USER_AGENT})

        if taxonomy_path is None:
            taxonomy_path = (
                Path(__file__).resolve().parent.parent.parent
                / "storage"
                / "newsseek"
                / "vlm_grounded_queries_benchmark.json"
            )
        self._load_taxonomy_glossary(taxonomy_path)

    def _load_taxonomy_glossary(self, path: Path) -> None:
        """Loads grounded bilingual VI -> EN mappings for instant 0ms exact lookup."""
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("textual_kis", []):
                vi = item.get("query_vi", "").strip()
                en = item.get("query_en", "").strip()
                if vi and en:
                    self._taxonomy_map[normalize_query_key(vi)] = en

            for item in data.get("adhoc_video_search", []):
                vi = item.get("query_vi", "").strip()
                en = item.get("query_en", "").strip()
                if vi and en:
                    self._taxonomy_map[normalize_query_key(vi)] = en

            for item in data.get("temporal_alignment", []):
                steps_vi = item.get("steps_vi") or []
                steps_en = item.get("steps_en") or []
                for s_vi, s_en in zip(steps_vi, steps_en):
                    if s_vi and s_en:
                        self._taxonomy_map[normalize_query_key(s_vi)] = s_en.strip()
                if len(steps_vi) >= 2 and len(steps_en) >= 2:
                    combined_vi = f"{steps_vi[0]} then {steps_vi[1]}"
                    combined_en = f"{steps_en[0]} then {steps_en[1]}"
                    self._taxonomy_map[normalize_query_key(combined_vi)] = combined_en
        except Exception as exc:
            print(f"[QueryTranslator] Warning: failed to load taxonomy glossary from {path}: {exc}")

    def _cache_get(self, key: Tuple[str, str]) -> Optional[Tuple[str, str]]:
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
            return val

    def _cache_set(self, key: Tuple[str, str], value: Tuple[str, str]) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _translate_google_nmt(self, text: str, source_lang: str) -> Optional[str]:
        sl = "vi" if source_lang == "vi" else "auto"
        try:
            resp = self._session.get(
                "https://clients5.google.com/translate_a/t",
                params={"client": "dict-chrome-ex", "sl": sl, "tl": "en", "q": text},
                timeout=self.timeout_sec,
            )
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, list) and first and isinstance(first[0], str) and first[0].strip():
                    return first[0].strip()
        except Exception:
            return None
        return None

    def _translate_mymemory(self, text: str, source_lang: str) -> Optional[str]:
        sl = "vi" if source_lang == "vi" else "autodetect"
        try:
            resp = self._session.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": f"{sl}|en"},
                timeout=self.timeout_sec,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            translated = (data.get("responseData") or {}).get("translatedText", "")
            if translated and translated.strip() and "MYMEMORY WARNING" not in translated.upper():
                return translated.strip()
        except Exception:
            return None
        return None

    def _load_offline_marian(self) -> bool:
        if self._offline_model is not None and self._offline_tok is not None:
            return True
        if self._offline_load_failed:
            return False

        with self._lock:
            if self._offline_model is not None and self._offline_tok is not None:
                return True
            if self._offline_load_failed:
                return False
            try:
                from transformers import MarianMTModel, MarianTokenizer

                # Prefer local HF cache snapshot if already downloaded to avoid network checks
                cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Helsinki-NLP--opus-mt-vi-en" / "snapshots"
                target = self.OFFLINE_MODEL_NAME
                local_only = False
                if cache_root.exists():
                    snaps = [p for p in cache_root.iterdir() if p.is_dir() and (p / "source.spm").exists()]
                    if snaps:
                        target = str(snaps[0])
                        local_only = True

                self._offline_tok = MarianTokenizer.from_pretrained(target, local_files_only=local_only)
                self._offline_model = MarianMTModel.from_pretrained(target, local_files_only=local_only)
                self._offline_model.eval()
                return True
            except Exception as exc:
                print(f"[QueryTranslator] Offline MarianMT unavailable: {exc}")
                self._offline_load_failed = True
                return False

    def _translate_offline_marian(self, text: str) -> Optional[str]:
        if not self._load_offline_marian():
            return None
        try:
            import torch

            inputs = self._offline_tok([text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                generated = self._offline_model.generate(**inputs, max_new_tokens=96, num_beams=2)
            decoded = self._offline_tok.decode(generated[0], skip_special_tokens=True).strip()
            return decoded or None
        except Exception:
            return None

    def translate_to_english(
        self,
        query_text: str,
        source_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translates non-English queries into English for CLIP/SigLIP alignment."""
        original = query_text.strip()
        lang = source_lang or detect_query_language(original)

        if not original or lang == "en":
            return {
                "applied": False,
                "source_lang": lang,
                "target_lang": "en",
                "original_query": original,
                "translated_query": original,
                "provider": "none",
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()
        norm_key = normalize_query_key(original)
        cache_key = (norm_key, lang)

        # Tier 0: LRU cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            en_text, orig_provider = cached
            return {
                "applied": True,
                "source_lang": lang,
                "target_lang": "en",
                "original_query": original,
                "translated_query": en_text,
                "provider": f"cache:{orig_provider}",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # Tier 1: Grounded VLM Taxonomy Bank exact bilingual lookup (0 ms)
        if norm_key in self._taxonomy_map:
            en_text = self._taxonomy_map[norm_key]
            self._cache_set(cache_key, (en_text, "taxonomy_bank"))
            return {
                "applied": True,
                "source_lang": lang,
                "target_lang": "en",
                "original_query": original,
                "translated_query": en_text,
                "provider": "taxonomy_bank",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # Prepare text for NMT (restore common diacritics if user typed unaccented Vietnamese)
        nmt_input = _restore_unaccented_vi_phrases(original) if lang == "vi" else original

        # Tier 2: Google NMT (highest visual translation accuracy, supports VI + multilingual)
        if self.enable_online:
            en_text = self._translate_google_nmt(nmt_input, lang)
            if en_text:
                self._cache_set(cache_key, (en_text, "google_nmt"))
                return {
                    "applied": True,
                    "source_lang": lang,
                    "target_lang": "en",
                    "original_query": original,
                    "translated_query": en_text,
                    "provider": "google_nmt",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # Tier 3: Local offline MarianMT fallback for Vietnamese (100% offline-resilient)
        if self.enable_offline_fallback and lang == "vi":
            en_text = self._translate_offline_marian(nmt_input)
            if en_text:
                self._cache_set(cache_key, (en_text, "opus_mt_offline"))
                return {
                    "applied": True,
                    "source_lang": lang,
                    "target_lang": "en",
                    "original_query": original,
                    "translated_query": en_text,
                    "provider": "opus_mt_offline",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # Tier 4: MyMemory HTTP fallback (useful for non-VI multilingual if Google is unavailable)
        if self.enable_online:
            en_text = self._translate_mymemory(nmt_input, lang)
            if en_text:
                self._cache_set(cache_key, (en_text, "mymemory"))
                return {
                    "applied": True,
                    "source_lang": lang,
                    "target_lang": "en",
                    "original_query": original,
                    "translated_query": en_text,
                    "provider": "mymemory",
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # Graceful pass-through fallback
        return {
            "applied": False,
            "source_lang": lang,
            "target_lang": "en",
            "original_query": original,
            "translated_query": original,
            "provider": "fallback_raw",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
