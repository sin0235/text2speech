from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import date as calendar_date
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf


class TTSError(RuntimeError):
    """Raised when the selected TTS engine cannot complete synthesis."""


TEXT_INPUT_LIMIT = 5000


GWEN_GENERATION_DEFAULTS: dict[str, Any] = {
    "speed": 1.0,
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "max_new_tokens": 4096,
    "repetition_penalty": 2.0,
    "subtalker_do_sample": True,
    "subtalker_temperature": 0.1,
    "subtalker_top_k": 20,
    "subtalker_top_p": 1.0,
    "subtalker_sampling_method": "gumbel",
}

GWEN_GENERATION_LIMITS: dict[str, dict[str, float | int]] = {
    "speed": {"min": 0.7, "max": 1.3, "step": 0.05},
    "temperature": {"min": 0.1, "max": 1.0, "step": 0.1},
    "top_k": {"min": 1, "max": 100, "step": 1},
    "top_p": {"min": 0.1, "max": 1.0, "step": 0.05},
    "max_new_tokens": {"min": 512, "max": 8192, "step": 256},
    "repetition_penalty": {"min": 1.0, "max": 3.0, "step": 0.1},
    "subtalker_temperature": {"min": 0.1, "max": 1.0, "step": 0.1},
    "subtalker_top_k": {"min": 1, "max": 100, "step": 1},
    "subtalker_top_p": {"min": 0.1, "max": 1.0, "step": 0.05},
}


def _clamp_float(value: Any, *, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return min(max(parsed, float(minimum)), float(maximum))


def _clamp_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return min(max(parsed, int(minimum)), int(maximum))


def _normalize_gwen_generation_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(GWEN_GENERATION_DEFAULTS)
    if raw:
        merged.update(raw)
    raw_subtalker_do_sample = merged.get("subtalker_do_sample", GWEN_GENERATION_DEFAULTS["subtalker_do_sample"])
    if isinstance(raw_subtalker_do_sample, str):
        normalized_subtalker_do_sample = raw_subtalker_do_sample.strip().lower() in {"1", "true", "yes", "on"}
    else:
        normalized_subtalker_do_sample = bool(raw_subtalker_do_sample)

    return {
        "speed": _clamp_float(
            merged.get("speed"),
            minimum=float(GWEN_GENERATION_LIMITS["speed"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["speed"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["speed"]),
        ),
        "temperature": _clamp_float(
            merged.get("temperature"),
            minimum=float(GWEN_GENERATION_LIMITS["temperature"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["temperature"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["temperature"]),
        ),
        "top_k": _clamp_int(
            merged.get("top_k"),
            minimum=int(GWEN_GENERATION_LIMITS["top_k"]["min"]),
            maximum=int(GWEN_GENERATION_LIMITS["top_k"]["max"]),
            default=int(GWEN_GENERATION_DEFAULTS["top_k"]),
        ),
        "top_p": _clamp_float(
            merged.get("top_p"),
            minimum=float(GWEN_GENERATION_LIMITS["top_p"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["top_p"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["top_p"]),
        ),
        "max_new_tokens": _clamp_int(
            merged.get("max_new_tokens"),
            minimum=int(GWEN_GENERATION_LIMITS["max_new_tokens"]["min"]),
            maximum=int(GWEN_GENERATION_LIMITS["max_new_tokens"]["max"]),
            default=int(GWEN_GENERATION_DEFAULTS["max_new_tokens"]),
        ),
        "repetition_penalty": _clamp_float(
            merged.get("repetition_penalty"),
            minimum=float(GWEN_GENERATION_LIMITS["repetition_penalty"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["repetition_penalty"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["repetition_penalty"]),
        ),
        "subtalker_do_sample": normalized_subtalker_do_sample,
        "subtalker_temperature": _clamp_float(
            merged.get("subtalker_temperature"),
            minimum=float(GWEN_GENERATION_LIMITS["subtalker_temperature"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["subtalker_temperature"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["subtalker_temperature"]),
        ),
        "subtalker_top_k": _clamp_int(
            merged.get("subtalker_top_k"),
            minimum=int(GWEN_GENERATION_LIMITS["subtalker_top_k"]["min"]),
            maximum=int(GWEN_GENERATION_LIMITS["subtalker_top_k"]["max"]),
            default=int(GWEN_GENERATION_DEFAULTS["subtalker_top_k"]),
        ),
        "subtalker_top_p": _clamp_float(
            merged.get("subtalker_top_p"),
            minimum=float(GWEN_GENERATION_LIMITS["subtalker_top_p"]["min"]),
            maximum=float(GWEN_GENERATION_LIMITS["subtalker_top_p"]["max"]),
            default=float(GWEN_GENERATION_DEFAULTS["subtalker_top_p"]),
        ),
        "subtalker_sampling_method": "gumbel",
    }


def _build_gwen_generation_kwargs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _normalize_gwen_generation_config(config)
    return {
        "speed": normalized["speed"],
        "temperature": normalized["temperature"],
        "top_k": normalized["top_k"],
        "top_p": normalized["top_p"],
        "max_new_tokens": normalized["max_new_tokens"],
        "repetition_penalty": normalized["repetition_penalty"],
        "subtalker_do_sample": normalized["subtalker_do_sample"],
        "subtalker_temperature": normalized["subtalker_temperature"],
        "subtalker_top_k": normalized["subtalker_top_k"],
        "subtalker_top_p": normalized["subtalker_top_p"],
    }


def _parse_pronunciation_overrides(
    sources: list[str] | None = None,
    targets: list[str] | None = None,
) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for source_raw, target_raw in zip(sources or [], targets or []):
        source = re.sub(r"\s+", " ", str(source_raw or "").strip())
        target = re.sub(r"\s+", " ", str(target_raw or "").strip())
        if not source or not target:
            continue
        parsed.append((source, target))
    return parsed


def _apply_pronunciation_overrides(
    text: str,
    overrides: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str, int]]]:
    if not overrides:
        return text, []

    updated_text = text
    applied: list[tuple[str, str, int]] = []
    for source, target in sorted(overrides, key=lambda item: len(item[0]), reverse=True):
        match_count = updated_text.count(source)
        if match_count <= 0:
            continue
        updated_text = updated_text.replace(source, target)
        applied.append((source, target, match_count))
    return updated_text, applied


_VI_DIGIT_WORDS = {
    0: "không",
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
}

_VI_BIG_UNITS = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]

_GWEN_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00a0", " "),
    ("…", "..."),
    ("–", ", "),
    ("—", ", "),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
)

_GWEN_ABBREVIATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bTP\.\s*HCM\b"), "thành phố hồ chí minh"),
    (re.compile(r"\bTP\.\s*HN\b"), "thành phố hà nội"),
    (re.compile(r"\bAI\b"), "ây ai"),
    (re.compile(r"\bKPI\b"), "cây pi ai"),
    (re.compile(r"\bCEO\b"), "si i ô"),
    (re.compile(r"\bCTO\b"), "si ti ô"),
    (re.compile(r"\bCFO\b"), "si ép ô"),
    (re.compile(r"\bCOO\b"), "si ô ô"),
    (re.compile(r"\bCMO\b"), "si em ô"),
    (re.compile(r"\bOK\b"), "ô kê"),
    (re.compile(r"\bID\b"), "ai đi"),
    (re.compile(r"\bSĐT\b"), "số điện thoại"),
    (re.compile(r"\bVNĐ\b|\bVND\b"), "đồng"),
    (re.compile(r"\bUSD\b"), "u ét đê"),
)

_GWEN_NUMBER_TOKEN_PATTERN = re.compile(
    r"(?<![\w/])([-+]?\d[\d.,]*)(\s?(?:%|đ|vnđ|vnd|usd|km|kg|gb|mb))?(?![\w/])",
    flags=re.IGNORECASE,
)

_GWEN_DMY_DATE_PATTERN = re.compile(
    r"(?<![\w])"
    r"(?P<day>0?[1-9]|[12]\d|3[01])"
    r"\s*(?P<sep>[./-])\s*"
    r"(?P<month>0?[1-9]|1[0-2])"
    r"\s*(?P=sep)\s*"
    r"(?P<year>\d{4})"
    r"(?![\w])",
    flags=re.IGNORECASE,
)

_GWEN_YMD_DATE_PATTERN = re.compile(
    r"(?<![\w])"
    r"(?P<year>\d{4})"
    r"\s*(?P<sep>[./-])\s*"
    r"(?P<month>0?[1-9]|1[0-2])"
    r"\s*(?P=sep)\s*"
    r"(?P<day>0?[1-9]|[12]\d|3[01])"
    r"(?![\w])",
    flags=re.IGNORECASE,
)

_GWEN_MONTH_YEAR_PATTERN = re.compile(
    r"(?<![\w])"
    r"(?P<month>0?[1-9]|1[0-2])"
    r"\s*(?P<sep>[./-])\s*"
    r"(?P<year>\d{4})"
    r"(?![\w])",
    flags=re.IGNORECASE,
)


def _read_vietnamese_triplet(number: int, *, force_hundreds: bool = False) -> str:
    if number <= 0:
        return "không" if force_hundreds else ""

    hundreds = number // 100
    tens_units = number % 100
    tens = tens_units // 10
    units = tens_units % 10
    parts: list[str] = []

    if hundreds or force_hundreds:
        parts.append(f"{_VI_DIGIT_WORDS.get(hundreds, 'không')} trăm")

    if tens > 1:
        parts.append(f"{_VI_DIGIT_WORDS[tens]} mươi")
        if units == 1:
            parts.append("mốt")
        elif units == 4:
            parts.append("tư")
        elif units == 5:
            parts.append("lăm")
        elif units > 0:
            parts.append(_VI_DIGIT_WORDS[units])
    elif tens == 1:
        parts.append("mười")
        if units == 5:
            parts.append("lăm")
        elif units > 0:
            parts.append(_VI_DIGIT_WORDS[units])
    elif units > 0:
        if hundreds or force_hundreds:
            parts.append("linh")
        parts.append(_VI_DIGIT_WORDS[units])

    return " ".join(part for part in parts if part).strip()


def _integer_to_vietnamese(number: int) -> str:
    if number == 0:
        return _VI_DIGIT_WORDS[0]
    if number < 0:
        return f"âm {_integer_to_vietnamese(abs(number))}"

    groups: list[int] = []
    while number > 0:
        groups.append(number % 1000)
        number //= 1000

    spoken_groups: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group_value = groups[index]
        if group_value == 0:
            continue
        higher_groups_present = any(value > 0 for value in groups[index + 1 :])
        force_hundreds = higher_groups_present and group_value < 100
        spoken = _read_vietnamese_triplet(group_value, force_hundreds=force_hundreds)
        unit = _VI_BIG_UNITS[index] if index < len(_VI_BIG_UNITS) else ""
        spoken_groups.append(" ".join(part for part in (spoken, unit) if part))

    return " ".join(spoken_groups).strip()


def _parse_number_token(raw_number: str) -> tuple[int | None, str | None]:
    token = raw_number.strip()
    if not token:
        return None, None

    sign = -1 if token.startswith("-") else 1
    token = token.lstrip("+-")
    if not token:
        return None, None

    separator_match = re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", token)
    if separator_match:
        return sign * int(token.replace(".", "").replace(",", "")), None

    if token.count(",") == 1 and token.count(".") == 0:
        left, right = token.split(",", 1)
        if left.isdigit() and right.isdigit():
            if len(right) == 3 and len(left) <= 3:
                return sign * int(left + right), None
            return sign * int(left), right

    if token.count(".") == 1 and token.count(",") == 0:
        left, right = token.split(".", 1)
        if left.isdigit() and right.isdigit():
            if len(right) == 3 and len(left) <= 3:
                return sign * int(left + right), None
            return sign * int(left), right

    compact = token.replace(".", "").replace(",", "")
    if compact.isdigit():
        return sign * int(compact), None
    return None, None


def _number_to_vietnamese_text(raw_number: str) -> str | None:
    integer_part, decimal_part = _parse_number_token(raw_number)
    if integer_part is None:
        return None

    spoken = _integer_to_vietnamese(integer_part)
    if decimal_part:
        decimal_words = " ".join(_VI_DIGIT_WORDS[int(char)] for char in decimal_part if char.isdigit())
        if decimal_words:
            spoken = f"{spoken} phẩy {decimal_words}"
    return spoken


def _context_ends_with_keyword(text: str, end_index: int, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\s*$", text[:end_index], flags=re.IGNORECASE) is not None


def _format_vietnamese_date(day: int, month: int, year: int, *, include_day_prefix: bool = True) -> str:
    day_text = _integer_to_vietnamese(day)
    month_text = _integer_to_vietnamese(month)
    year_text = _integer_to_vietnamese(year)
    if include_day_prefix:
        return f"ngày {day_text} tháng {month_text} năm {year_text}"
    return f"{day_text} tháng {month_text} năm {year_text}"


def _format_vietnamese_month_year(month: int, year: int, *, include_month_prefix: bool = True) -> str:
    month_text = _integer_to_vietnamese(month)
    year_text = _integer_to_vietnamese(year)
    if include_month_prefix:
        return f"tháng {month_text} năm {year_text}"
    return f"{month_text} năm {year_text}"


def _normalize_gwen_date_tokens(text: str) -> tuple[str, int]:
    normalized = text
    date_hits = 0

    def replace_dmy(match: re.Match[str]) -> str:
        nonlocal date_hits
        day = int(match.group("day"))
        month = int(match.group("month"))
        year = int(match.group("year"))
        try:
            calendar_date(year, month, day)
        except ValueError:
            return match.group(0)
        date_hits += 1
        include_day_prefix = not (
            _context_ends_with_keyword(normalized, match.start(), "ngày")
            or _context_ends_with_keyword(normalized, match.start(), "mùng")
        )
        return _format_vietnamese_date(day, month, year, include_day_prefix=include_day_prefix)

    normalized = _GWEN_DMY_DATE_PATTERN.sub(replace_dmy, normalized)

    def replace_ymd(match: re.Match[str]) -> str:
        nonlocal date_hits
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        try:
            calendar_date(year, month, day)
        except ValueError:
            return match.group(0)
        date_hits += 1
        include_day_prefix = not (
            _context_ends_with_keyword(normalized, match.start(), "ngày")
            or _context_ends_with_keyword(normalized, match.start(), "mùng")
        )
        return _format_vietnamese_date(day, month, year, include_day_prefix=include_day_prefix)

    normalized = _GWEN_YMD_DATE_PATTERN.sub(replace_ymd, normalized)

    def replace_month_year(match: re.Match[str]) -> str:
        nonlocal date_hits
        month = int(match.group("month"))
        year = int(match.group("year"))
        if month < 1 or month > 12:
            return match.group(0)
        date_hits += 1
        include_month_prefix = not _context_ends_with_keyword(normalized, match.start(), "tháng")
        return _format_vietnamese_month_year(month, year, include_month_prefix=include_month_prefix)

    normalized = _GWEN_MONTH_YEAR_PATTERN.sub(replace_month_year, normalized)
    return normalized, date_hits


def _strip_sea_language_tags(text: str) -> str:
    return re.sub(r"</?en>", "", text)


def _get_sea_normalizer() -> Any | None:
    try:
        from sea_g2p import Normalizer
        return Normalizer(lang="vi")
    except Exception:
        return None


_sea_normalizer_instance: Any | None = None
_sea_normalizer_loaded = False


def _normalize_gwen_text(text: str) -> tuple[str, list[str]]:
    global _sea_normalizer_instance, _sea_normalizer_loaded
    normalized = unicodedata.normalize("NFC", text or "")
    notes: list[str] = []

    if not _sea_normalizer_loaded:
        _sea_normalizer_instance = _get_sea_normalizer()
        _sea_normalizer_loaded = True

    if _sea_normalizer_instance is not None:
        try:
            sea_result = _sea_normalizer_instance.normalize(normalized)
            if isinstance(sea_result, list):
                sea_result = sea_result[0] if sea_result else normalized
            normalized = _strip_sea_language_tags(str(sea_result))
            notes.append("Đã chuẩn hóa văn bản bằng SEA-G2P Normalizer (số, ngày, viết tắt, email, tiền tệ).")
        except Exception:
            normalized = _normalize_gwen_text_fallback(normalized, notes)
    else:
        normalized = _normalize_gwen_text_fallback(normalized, notes)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;!?])([^\s])", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, notes


def _normalize_gwen_text_fallback(text: str, notes: list[str]) -> str:
    normalized = text

    for source, target in _GWEN_TEXT_REPLACEMENTS:
        if source in normalized:
            normalized = normalized.replace(source, target)

    normalized, date_hits = _normalize_gwen_date_tokens(normalized)

    abbreviation_hits = 0
    for pattern, replacement in _GWEN_ABBREVIATION_PATTERNS:
        normalized, replaced = pattern.subn(replacement, normalized)
        abbreviation_hits += replaced

    number_hits = 0

    def replace_number(match: re.Match[str]) -> str:
        nonlocal number_hits
        number_raw = match.group(1)
        suffix_raw = (match.group(2) or "").strip().lower()
        spoken_number = _number_to_vietnamese_text(number_raw)
        if not spoken_number:
            return match.group(0)

        unit_map = {
            "%": "phần trăm",
            "đ": "đồng",
            "vnđ": "đồng",
            "vnd": "đồng",
            "usd": "u ét đê",
            "km": "ki lô mét",
            "kg": "ki lô gam",
            "gb": "gi ga bai",
            "mb": "mê ga bai",
        }
        suffix_text = unit_map.get(suffix_raw, suffix_raw)
        number_hits += 1
        return " ".join(part for part in (spoken_number, suffix_text) if part).strip()

    normalized = _GWEN_NUMBER_TOKEN_PATTERN.sub(replace_number, normalized)

    if date_hits:
        notes.append("Đã chuẩn hóa ngày tháng năm viết bằng số sang cách đọc tiếng Việt tự nhiên hơn.")
    if abbreviation_hits:
        notes.append("Đã chuẩn hóa chữ viết tắt tiếng Việt/Latin sang cách đọc tự nhiên hơn.")
    if number_hits:
        notes.append("Đã chuẩn hóa số và đơn vị sang dạng đọc tiếng Việt trước khi synthesize.")
    return normalized


def _summarize_gwen_generation_changes(config: dict[str, Any] | None = None) -> str | None:
    normalized = _normalize_gwen_generation_config(config)
    changes: list[str] = []
    for key, label, formatter in (
        ("speed", "speed", lambda value: f"{float(value):.2f}x"),
        ("temperature", "temperature", lambda value: f"{float(value):.1f}"),
        ("top_p", "top_p", lambda value: f"{float(value):.2f}"),
        ("top_k", "top_k", lambda value: str(int(value))),
        ("repetition_penalty", "repetition_penalty", lambda value: f"{float(value):.1f}"),
        ("max_new_tokens", "max_new_tokens", lambda value: str(int(value))),
        ("subtalker_do_sample", "cp_do_sample", lambda value: "on" if value else "off"),
        ("subtalker_temperature", "cp_temperature", lambda value: f"{float(value):.1f}"),
        ("subtalker_top_k", "cp_top_k", lambda value: str(int(value))),
        ("subtalker_top_p", "cp_top_p", lambda value: f"{float(value):.2f}"),
    ):
        if normalized[key] != GWEN_GENERATION_DEFAULTS[key]:
            changes.append(f"{label}={formatter(normalized[key])}")
    if not changes:
        return None
    return ", ".join(changes)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _nonempty_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and any(path.iterdir())
    except Exception:
        return False


def _parse_named_choices(raw: str) -> list[tuple[str, str]]:
    if not raw:
        return []

    pairs: list[tuple[str, str]] = []
    for item in re.split(r"[;\r\n]+", raw):
        cleaned = item.strip().strip(",")
        if not cleaned:
            continue
        if "=" in cleaned:
            label, value = cleaned.split("=", 1)
        elif "|" in cleaned:
            label, value = cleaned.split("|", 1)
        else:
            label, value = cleaned, cleaned
        label = label.strip() or value.strip()
        value = value.strip()
        if value:
            pairs.append((label, value))
    return pairs



def _sanitize_cache_token(value: str, fallback: str = "default") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip()).strip("-._")
    return cleaned[:80] or fallback


def _map_import_name_to_package(module_name: str) -> str:
    return {
        "hydra": "hydra-core",
        "cached_path": "cached_path",
        "flash_attn": "flash-attn",
        "llama_cpp": "llama-cpp-python",
        "omegaconf": "omegaconf",
        "onnxruntime": "onnxruntime",
        "pydub": "pydub",
        "perth": "perth",
        "pypinyin": "pypinyin",
        "qwen_tts": "qwen-tts",
        "rjieba": "rjieba",
        "sea_g2p": "sea-g2p",
        "sentencepiece": "sentencepiece",
        "transformers": "transformers",
        "transformers_stream_generator": "transformers-stream-generator",
        "unidecode": "unidecode",
        "x_transformers": "x-transformers",
    }.get((module_name or "").strip(), (module_name or "").strip())




def _format_gwen_import_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        missing_module = exc.name.strip()
        package_name = _map_import_name_to_package(missing_module)
        return (
            f"Gwen-TTS đã được cài nhưng thiếu dependency `{missing_module}`. "
            f"Hãy chạy `python -m pip install -U qwen-tts {package_name}` rồi khởi động lại ứng dụng."
        )
    return f"Không thể import Gwen-TTS: {exc}"


def _format_gwen_runtime_error(exc: Exception) -> str:
    normalized = re.sub(r"\s+", " ", str(exc or "")).strip()
    lowered = normalized.lower()
    if "flash_attn" in lowered or "flash attention" in lowered:
        return (
            "Gwen-TTS không khởi tạo được với FlashAttention trong runtime hiện tại. "
            "App đã có thể fallback sang `sdpa`, nhưng nếu vẫn lỗi hãy cài `flash-attn` "
            "hoặc đặt `GWEN_ATTN_IMPLEMENTATION=sdpa` rồi khởi động lại."
        )
    if "out of memory" in lowered or "cuda error" in lowered:
        return (
            "Gwen-TTS bị lỗi GPU/CUDA khi load hoặc sinh audio. "
            "Hãy kiểm tra VRAM còn trống, giảm tải runtime, hoặc dùng GPU Colab mạnh hơn."
        )
    if "cuda" in lowered and "available" in lowered:
        return "Gwen-TTS trong project này cần GPU CUDA để chạy local."
    return normalized


def _format_asr_import_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        missing_module = exc.name.strip()
        package_name = _map_import_name_to_package(missing_module)
        return (
            f"Chức năng nhận diện transcript đã được bật nhưng thiếu dependency `{missing_module}`. "
            f"Hãy chạy `python -m pip install -U transformers accelerate sentencepiece {package_name}` "
            "rồi khởi động lại ứng dụng."
        )
    return f"Không thể import module nhận diện transcript: {exc}"


def _format_asr_runtime_error(exc: Exception) -> str:
    normalized = re.sub(r"\s+", " ", str(exc or "")).strip()
    lowered = normalized.lower()
    if "out of memory" in lowered or "cuda error" in lowered:
        return (
            "Model nhận diện transcript bị lỗi GPU/CUDA khi load hoặc infer. "
            "Hãy giảm model ASR, giải phóng VRAM, hoặc cho ASR chạy trên CPU."
        )
    return f"Không thể nhận diện transcript từ audio tham chiếu: {normalized}"



def _split_text_for_tts(text: str, max_chars: int = 220) -> list[str]:
    """Split long Vietnamese text into short sentences/chunks for stable inference."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+", normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        words = sentence.split()
        part = ""
        for word in words:
            attempt = word if not part else f"{part} {word}"
            if len(attempt) <= max_chars:
                part = attempt
            else:
                if part:
                    chunks.append(part)
                part = word
        if part:
            current = part

    if current:
        chunks.append(current)
    return chunks


def _normalize_transcription_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return ""
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([(\[{])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([)\]}])", r"\1", normalized)
    return normalized


def _normalize_tts_prompt_text(text: str) -> tuple[str, list[str]]:
    normalized = unicodedata.normalize("NFC", text or "")
    notes: list[str] = []

    invisible_hits = 0
    for source, target in (
        ("\u00a0", " "),
        ("\u200b", ""),
        ("\u200c", ""),
        ("\u200d", ""),
        ("\ufeff", ""),
    ):
        if source in normalized:
            normalized = normalized.replace(source, target)
            invisible_hits += 1

    normalized, ellipsis_hits = re.subn(r"(?:\.{3,}|…)", ", ", normalized)
    normalized, dash_hits = re.subn(r"[ \t]+[-–—]{1,}[ \t]+", ", ", normalized)

    line_break_hits = 0
    bullet_hits = 0
    if re.search(r"\r?\n", normalized):
        segments: list[str] = []
        for raw_segment in re.split(r"(?:\r?\n)+", normalized):
            segment = raw_segment.strip()
            if not segment:
                continue
            cleaned_segment, removed = re.subn(r"^[\-*+•▪‣◦·]+\s*", "", segment)
            bullet_hits += removed
            cleaned_segment = re.sub(r"\s+", " ", cleaned_segment).strip(" ,")
            if cleaned_segment:
                segments.append(cleaned_segment)
        if segments:
            rebuilt = segments[0]
            for segment in segments[1:]:
                separator = " " if rebuilt and rebuilt[-1] in ".!?;:" else ". "
                rebuilt = f"{rebuilt}{separator}{segment}"
            normalized = rebuilt
            line_break_hits = 1

    normalized, repeated_punct_hits = re.subn(r"([,;:!?])\1+", r"\1", normalized)
    normalized, repeated_stop_hits = re.subn(r"\.{2,}", ".", normalized)

    before_spacing = normalized
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,;:!?])(?=[^\s])", r"\1 ", normalized)
    normalized = re.sub(r"(?<!\d)\.(?=[^\s\d])", ". ", normalized)
    normalized = re.sub(r"([(\[{])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([)\]}])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,")
    spacing_changed = normalized != before_spacing

    terminal_hits = 0
    if normalized and len(normalized) >= 18 and re.search(r"[.!?][\"')\]}]*$", normalized) is None:
        normalized += "."
        terminal_hits = 1

    if line_break_hits or bullet_hits:
        notes.append("Đã chuẩn hóa xuống dòng và danh sách thành nhịp câu liền mạch hơn.")
    if ellipsis_hits or dash_hits or repeated_punct_hits or repeated_stop_hits:
        notes.append("Đã dọn dấu ngắt, dấu chấm lửng và dấu câu lặp để giọng đọc trôi chảy hơn.")
    if invisible_hits or spacing_changed or terminal_hits:
        notes.append("Đã chuẩn hóa khoảng trắng và điểm dừng cuối câu trước khi synthesize.")
    return normalized, notes


def _split_text_by_words(text: str, *, max_words: int, max_chars: int | None = None) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= max_words and (max_chars is None or len(normalized) <= max_chars):
        return [normalized]

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)
        if current_words and (len(candidate_words) > max_words or (max_chars is not None and len(candidate) > max_chars)):
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words = candidate_words

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks




def _crossfade_join(waves: list[np.ndarray], sample_rate: int, duration_ms: int = 80) -> np.ndarray:
    if not waves:
        return np.zeros(1, dtype=np.float32)
    if len(waves) == 1:
        return waves[0]

    crossfade = max(0, int(sample_rate * duration_ms / 1000))
    output = waves[0].astype(np.float32)

    for wave in waves[1:]:
        current = wave.astype(np.float32)
        if crossfade <= 0 or len(output) <= crossfade or len(current) <= crossfade:
            output = np.concatenate([output, current])
            continue

        fade_out = np.linspace(1.0, 0.0, crossfade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        mixed_tail = output[-crossfade:] * fade_out + current[:crossfade] * fade_in
        output = np.concatenate([output[:-crossfade], mixed_tail, current[crossfade:]])

    return output


def _to_numpy_audio(audio: Any) -> np.ndarray:
    tensor = audio
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor, dtype=np.float32).reshape(-1)


def _numel(obj: Any) -> int | None:
    if hasattr(obj, "numel"):
        try:
            return int(obj.numel())
        except Exception:
            return None
    if hasattr(obj, "size"):
        try:
            size = obj.size
            if isinstance(size, int):
                return int(size)
        except Exception:
            return None
    return None


def _normalize_reference_wave(audio_np: np.ndarray, target_peak: float = 0.92) -> np.ndarray:
    normalized = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if normalized.size == 0:
        return normalized

    normalized = normalized - float(np.mean(normalized))
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak <= 1e-6:
        raise TTSError("Audio tham chiếu gần như im lặng. Hãy dùng file có tiếng nói rõ hơn.")

    normalized = normalized / peak
    return np.clip(normalized * float(target_peak), -1.0, 1.0).astype(np.float32)


def _summarize_subprocess_error(exc: Exception) -> str:
    stderr = getattr(exc, "stderr", None)
    stdout = getattr(exc, "stdout", None)
    raw = stderr or stdout or str(exc)
    normalized = re.sub(r"\s+", " ", str(raw or "")).strip()
    return normalized[:280] or str(exc)


def _load_audio_mono_float(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
    except Exception:
        try:
            import librosa
        except Exception as exc:
            raise TTSError(
                "Không đọc được audio tham chiếu. Hãy upload WAV/FLAC hoặc cài `librosa` để hỗ trợ MP3/M4A."
            ) from exc
        audio_np, sr = librosa.load(str(path), sr=None, mono=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)

    if audio_np.ndim == 0:
        raise TTSError("Audio tham chiếu rỗng hoặc không hợp lệ.")

    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=-1 if audio_np.shape[-1] <= 8 else 0)

    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0:
        raise TTSError("Audio tham chiếu không chứa sample hợp lệ.")

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak <= 1e-6:
        raise TTSError("Audio tham chiếu gần như im lặng. Hãy dùng file có tiếng nói rõ hơn.")

    if sr != target_sr:
        try:
            import librosa
        except Exception as exc:
            raise TTSError(
                f"Audio tham chiếu đang ở {sr} Hz. Cần cài `librosa` để resample về {target_sr} Hz cho engine hiện tại."
            ) from exc
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio_np = _normalize_reference_wave(audio_np)
    return audio_np, sr


def _inspect_audio_mono_float(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
    except Exception:
        try:
            import librosa
        except Exception as exc:
            raise TTSError(
                "Không đọc được audio tham chiếu. Hãy upload WAV/FLAC hoặc cài `librosa` để hỗ trợ MP3/M4A."
            ) from exc
        audio_np, sr = librosa.load(str(path), sr=None, mono=False)
        audio_np = np.asarray(audio_np, dtype=np.float32)

    if audio_np.ndim == 0:
        raise TTSError("Audio tham chiếu rỗng hoặc không hợp lệ.")

    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=-1 if audio_np.shape[-1] <= 8 else 0)

    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0:
        raise TTSError("Audio tham chiếu không chứa sample hợp lệ.")

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak <= 1e-6:
        raise TTSError("Audio tham chiếu gần như im lặng. Hãy dùng file có tiếng nói rõ hơn.")

    return audio_np, int(sr)


def _estimate_activity_ratio(audio_np: np.ndarray, sample_rate: int) -> float:
    if audio_np.size == 0 or sample_rate <= 0:
        return 0.0

    window = min(len(audio_np), max(1, int(sample_rate * 0.03)))
    envelope = np.convolve(
        np.abs(audio_np),
        np.ones(window, dtype=np.float32) / float(window),
        mode="same",
    )
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak <= 1e-6:
        return 0.0

    threshold = max(0.003, peak * 0.02)
    return float(np.mean(envelope >= threshold))


def _trim_reference_silence(audio_np: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, float | bool]]:
    if audio_np.size == 0 or sample_rate <= 0:
        return audio_np, {
            "trimmed": False,
            "trimmed_seconds": 0.0,
            "original_duration_seconds": 0.0,
            "activity_ratio": 0.0,
        }

    window = min(len(audio_np), max(1, int(sample_rate * 0.03)))
    envelope = np.convolve(
        np.abs(audio_np),
        np.ones(window, dtype=np.float32) / float(window),
        mode="same",
    )
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak <= 1e-6:
        return audio_np, {
            "trimmed": False,
            "trimmed_seconds": 0.0,
            "original_duration_seconds": float(len(audio_np) / sample_rate),
            "activity_ratio": 0.0,
        }

    threshold = max(0.003, peak * 0.02)
    active = envelope >= threshold
    activity_ratio = float(np.mean(active)) if active.size else 0.0
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        return audio_np, {
            "trimmed": False,
            "trimmed_seconds": 0.0,
            "original_duration_seconds": float(len(audio_np) / sample_rate),
            "activity_ratio": activity_ratio,
        }

    padding = int(sample_rate * 0.08)
    start = max(0, int(active_idx[0]) - padding)
    end = min(len(audio_np), int(active_idx[-1]) + padding + 1)
    trimmed = audio_np[start:end].astype(np.float32, copy=False)
    trimmed_seconds = max(0.0, float((len(audio_np) - len(trimmed)) / sample_rate))

    return trimmed, {
        "trimmed": bool(start > 0 or end < len(audio_np)),
        "trimmed_seconds": trimmed_seconds,
        "original_duration_seconds": float(len(audio_np) / sample_rate),
        "activity_ratio": activity_ratio,
    }


@dataclass(slots=True)
class EngineCard:
    id: str
    label: str
    headline: str
    description: str
    recommended_for: str
    output_quality: str
    reference_hint: str
    supports_reference_text: bool
    ready: bool
    summary: str
    warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PresetVoice:
    id: str
    name: str
    avatar: str
    style: str
    audio_filename: str
    reference_text: str


@dataclass(slots=True)
class SynthesisResult:
    engine_id: str
    engine_label: str
    model_key: str
    model_label: str
    output_path: Path
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    chunk_count: int
    reference_text_used: bool
    seed: int | None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str
    model_id: str
    duration_seconds: float
    inference_seconds: float
    notes: list[str] = field(default_factory=list)


class TTSStudioService:
    """Flask-facing service for Gwen-TTS voice cloning."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runtime_dir = self.root / "webapp" / "runtime"
        self.reference_dir = self.runtime_dir / "references"
        self.output_dir = self.runtime_dir / "generated"
        self.model_dir = self.root / "models"
        self.static_dir = self.root / "webapp" / "static"
        self.voice_preset_dir = self.static_dir / "voice_presets"
        self.gwen_preset_config_path = self.root / "webapp" / "data" / "gwen_preset_voices.json"
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.default_engine = "gwen"
        self.gwen_model_id = os.getenv("GWEN_MODEL_ID", "g-group-ai-lab/gwen-tts-0.6B").strip() or "g-group-ai-lab/gwen-tts-0.6B"
        self.gwen_model_choices = os.getenv("GWEN_MODEL_CHOICES", "").strip()
        self.gwen_dtype = (os.getenv("GWEN_DTYPE", "bfloat16") or "bfloat16").strip().lower() or "bfloat16"
        self.gwen_attn_implementation = (os.getenv("GWEN_ATTN_IMPLEMENTATION", "flash_attention_2") or "flash_attention_2").strip() or "flash_attention_2"
        self.asr_model_id = os.getenv("ASR_MODEL_ID", "openai/whisper-small").strip() or "openai/whisper-small"
        self.asr_language = (os.getenv("ASR_LANGUAGE", "vi") or "vi").strip().lower() or "vi"
        self.asr_chunk_length_s = _clamp_int(os.getenv("ASR_CHUNK_LENGTH_S", "18"), minimum=8, maximum=30, default=18)
        self._locks = {
            "gwen": Lock(),
            "asr": Lock(),
        }
        self._loaded_models: dict[str, Any] = {}
        self._gwen_import_probe: tuple[bool, str | None] | None = None
        self._asr_import_probe: tuple[bool, str | None] | None = None
        self._gwen_preset_voices_cache: list[PresetVoice] | None = None
        self._sea_normalizer: Any | None = None
        self._sea_normalizer_checked = False

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(path)

    @staticmethod
    def _make_gwen_model_key(model_id: str) -> str:
        return f"model::{model_id.strip()}"

    def _load_gwen_preset_voices(self) -> list[PresetVoice]:
        if self._gwen_preset_voices_cache is not None:
            return self._gwen_preset_voices_cache

        if not self.gwen_preset_config_path.exists():
            self._gwen_preset_voices_cache = []
            return self._gwen_preset_voices_cache

        try:
            raw_items = json.loads(self.gwen_preset_config_path.read_text(encoding="utf-8"))
        except Exception:
            self._gwen_preset_voices_cache = []
            return self._gwen_preset_voices_cache

        voices: list[PresetVoice] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            voice_id = str(item.get("id", "")).strip()
            if not voice_id:
                continue
            voices.append(
                PresetVoice(
                    id=voice_id,
                    name=str(item.get("name", voice_id)).strip() or voice_id,
                    avatar=str(item.get("avatar", voice_id[:2].upper())).strip() or voice_id[:2].upper(),
                    style=str(item.get("style", "Preset")).strip() or "Preset",
                    audio_filename=str(item.get("audio_filename", f"{voice_id}.wav")).strip() or f"{voice_id}.wav",
                    reference_text=str(item.get("reference_text", "")).strip(),
                )
            )

        self._gwen_preset_voices_cache = voices
        return self._gwen_preset_voices_cache

    def get_preset_voices(self, engine_id: str = "gwen") -> list[PresetVoice]:
        return self._load_gwen_preset_voices()

    def get_preset_voice(self, engine_id: str, preset_voice_id: str) -> PresetVoice:

        normalized_id = (preset_voice_id or "").strip().lower()
        for voice in self._load_gwen_preset_voices():
            if voice.id == normalized_id:
                return voice
        raise TTSError(f"Không tìm thấy preset voice '{preset_voice_id}'.")

    def get_preset_voice_reference(self, engine_id: str, preset_voice_id: str) -> tuple[PresetVoice, Path]:
        voice = self.get_preset_voice(engine_id, preset_voice_id)
        audio_path = self.voice_preset_dir / voice.audio_filename
        if not audio_path.exists():
            raise TTSError(
                f"Không tìm thấy file audio của preset voice '{voice.name}' tại '{self._display_path(audio_path)}'."
            )
        return voice, audio_path

    def _probe_gwen_import(self) -> tuple[bool, str | None]:
        if self._gwen_import_probe is not None:
            return self._gwen_import_probe

        if importlib.util.find_spec("qwen_tts") is None:
            self._gwen_import_probe = (False, "Chưa cài gói `qwen_tts`.")
            return self._gwen_import_probe

        try:
            importlib.import_module("qwen_tts")
        except Exception as exc:
            self._gwen_import_probe = (False, _format_gwen_import_error(exc))
            return self._gwen_import_probe

        self._gwen_import_probe = (True, None)
        return self._gwen_import_probe

    def _probe_asr_import(self) -> tuple[bool, str | None]:
        if self._asr_import_probe is not None:
            return self._asr_import_probe

        if importlib.util.find_spec("transformers") is None:
            self._asr_import_probe = (
                False,
                "Chưa cài `transformers`. Hãy chạy `python -m pip install -U transformers accelerate sentencepiece` để bật nhận diện transcript.",
            )
            return self._asr_import_probe

        try:
            importlib.import_module("transformers")
        except Exception as exc:
            self._asr_import_probe = (False, _format_asr_import_error(exc))
            return self._asr_import_probe

        self._asr_import_probe = (True, None)
        return self._asr_import_probe

    def _gwen_model_selection(self) -> dict[str, Any]:
        options = [
            {
                "key": "default",
                "label": f"Mặc định: {self.gwen_model_id}",
                "value": self.gwen_model_id,
                "mode": "default",
                "requires_reference_text": True,
            }
        ]
        seen_keys = {"default"}
        for label, value in _parse_named_choices(self.gwen_model_choices):
            key = self._make_gwen_model_key(value)
            if key in seen_keys:
                continue
            options.append(
                {
                    "key": key,
                    "label": label,
                    "value": value,
                    "mode": "name",
                    "requires_reference_text": True,
                }
            )
            seen_keys.add(key)

        return {
            "default_key": "default",
            "allow_custom": True,
            "default_value": self.gwen_model_id,
            "custom_placeholder": "Ví dụ: g-group-ai-lab/gwen-tts-0.6B hoặc repo HF khác tương thích qwen-tts",
            "help_primary": "Gwen-TTS đang là model chủ chốt của project. Luôn cần transcript tham chiếu chính xác.",
            "help_secondary": "Model được nạp qua `qwen_tts.Qwen3TTSModel.from_pretrained(...)`; custom cho phép đổi sang repo HF compatible khác.",
            "options": options,
        }

    def get_model_selection(self, engine_id: str) -> dict[str, Any]:
        return self._gwen_model_selection()

    def resolve_model_spec(
        self,
        engine_id: str = "gwen",
        *,
        model_key: str = "default",
        custom_model: str = "",
    ) -> dict[str, Any]:
        key = (model_key or "default").strip() or "default"
        custom = (custom_model or "").strip()

        if key == "__custom__":
            if not custom:
                raise TTSError("Hãy nhập model Gwen tùy chỉnh trước khi generate.")
            model_id = custom
        elif key == "default":
            return {
                "key": "default",
                "label": self.gwen_model_id,
                "mode": "default",
                "model_id": self.gwen_model_id,
                "dtype": self.gwen_dtype,
                "attn_implementation": self.gwen_attn_implementation,
                "cache_key": f"gwen::default::{self.gwen_model_id}::{self.gwen_dtype}::{self.gwen_attn_implementation}",
            }
        elif key.startswith("model::"):
            model_id = key.split("::", 1)[1].strip()
        else:
            raise TTSError("Lựa chọn model Gwen không hợp lệ.")
        if not model_id:
            raise TTSError("Model Gwen không hợp lệ.")
        return {
            "key": self._make_gwen_model_key(model_id),
            "label": model_id,
            "mode": "name",
            "model_id": model_id,
            "dtype": self.gwen_dtype,
            "attn_implementation": self.gwen_attn_implementation,
            "cache_key": f"gwen::{model_id}::{self.gwen_dtype}::{self.gwen_attn_implementation}",
        }

    def summary(self) -> dict[str, Any]:
        card = self._gwen_card()
        device_label = "GPU" if self._torch_cuda_available() else "CPU"
        return {
            "engine_count": 1,
            "ready_count": 1 if card.ready else 0,
            "device_label": device_label,
            "default_engine": card.label,
        }

    def get_engine_cards(self) -> list[EngineCard]:
        return [self._gwen_card()]

    def get_engine_card(self, engine_id: str = "gwen") -> EngineCard:
        return self._gwen_card()

    def synthesize(
        self,
        *,
        engine_id: str = "gwen",
        text: str,
        reference_audio: Path,
        reference_text: str = "",
        speed: float = 1.0,
        remove_silence: bool = False,
        seed: int | None = None,
        model_key: str = "default",
        custom_model: str = "",
        gwen_generation_config: dict[str, Any] | None = None,
        pronunciation_overrides: list[tuple[str, str]] | None = None,
    ) -> SynthesisResult:
        text = (text or "").strip()
        reference_text = re.sub(r"\s+", " ", (reference_text or "").strip())

        if not text:
            raise TTSError("Thiếu nội dung văn bản cần chuyển giọng nói.")
        if len(re.sub(r"\s+", " ", text)) > TEXT_INPUT_LIMIT:
            raise TTSError(f"Văn bản quá dài. Giới hạn hiện tại là {TEXT_INPUT_LIMIT} ký tự mỗi lượt.")
        if not reference_audio.exists():
            raise TTSError("Không tìm thấy file audio tham chiếu.")

        text, input_normalization_notes = _normalize_tts_prompt_text(text)
        if not text:
            raise TTSError("Văn bản đầu vào rỗng sau khi chuẩn hóa cho TTS.")

        engine = self.get_engine_card()
        if not engine.ready:
            raise TTSError(engine.warning or engine.summary)
        model_spec = self.resolve_model_spec(model_key=model_key, custom_model=custom_model)
        if not reference_text:
            raise TTSError(
                "Gwen-TTS cần transcript tham chiếu. Hãy nhập đúng câu đang có trong audio tham chiếu trước khi generate."
            )

        output_path = self.output_dir / f"{int(time.time())}-{uuid.uuid4().hex[:10]}-gwen.wav"
        result = self._synthesize_with_gwen(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            model_spec=model_spec,
            output_path=output_path,
            speed=speed,
            gwen_generation_config=gwen_generation_config,
            pronunciation_overrides=pronunciation_overrides,
        )

        if input_normalization_notes:
            result.notes.insert(0, " ".join(input_normalization_notes))
        return result

    def save_reference_file(self, filename: str, payload: bytes) -> Path:
        extension = Path(filename).suffix.lower() or ".wav"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", Path(filename).stem).strip("-") or "reference"
        saved = self.reference_dir / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{safe_name}{extension}"
        saved.write_bytes(payload)
        return saved

    def _load_asr(self) -> dict[str, Any]:
        cache_key = f"asr::{self.asr_model_id}"
        with self._locks["asr"]:
            cached = self._loaded_models.get(cache_key)
            if cached is not None:
                return cached

            module_ok, warning = self._probe_asr_import()
            if not module_ok:
                raise TTSError(warning or "Chưa cài module nhận diện transcript.")

            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            except Exception as exc:
                raise TTSError(_format_asr_import_error(exc)) from exc

            try:
                use_cuda = bool(torch.cuda.is_available())
                torch_dtype = torch.float16 if use_cuda else torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.asr_model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                processor = AutoProcessor.from_pretrained(self.asr_model_id)
                asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=0 if use_cuda else -1,
                    chunk_length_s=self.asr_chunk_length_s,
                )
            except Exception as exc:
                raise TTSError(_format_asr_runtime_error(exc)) from exc

            loaded = {
                "pipeline": asr_pipeline,
                "model_id": self.asr_model_id,
                "language": self.asr_language,
                "device_label": "cuda:0" if use_cuda else "cpu",
            }
            self._loaded_models[cache_key] = loaded
            return loaded

    def transcribe_reference_audio(self, reference_audio: Path) -> TranscriptionResult:
        reference_audio = Path(reference_audio)
        if not reference_audio.exists():
            raise TTSError("Không tìm thấy file audio để nhận diện transcript.")

        audio_np, sample_rate = _load_audio_mono_float(reference_audio, target_sr=16000)
        audio_np, prep_stats = _trim_reference_silence(audio_np, sample_rate)
        duration_seconds = float(len(audio_np) / sample_rate) if sample_rate > 0 else 0.0
        if duration_seconds < 0.35:
            raise TTSError("Audio tham chiếu quá ngắn để nhận diện transcript. Hãy dùng đoạn nói rõ tối thiểu khoảng nửa giây.")

        loaded = self._load_asr()
        asr_pipeline = loaded["pipeline"]
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if self.asr_language:
            generate_kwargs["language"] = self.asr_language

        started = time.perf_counter()
        try:
            response = asr_pipeline(
                {"array": np.asarray(audio_np, dtype=np.float32), "sampling_rate": sample_rate},
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
        except Exception as exc:
            raise TTSError(_format_asr_runtime_error(exc)) from exc
        elapsed = time.perf_counter() - started

        raw_text = response.get("text", "") if isinstance(response, dict) else str(response or "")
        text = _normalize_transcription_text(raw_text)
        if not text:
            raise TTSError("Model nhận diện không trả về transcript nào. Hãy thử audio rõ tiếng hơn hoặc đổi model ASR.")

        notes = [
            f"ASR model: {loaded['model_id']}.",
            f"Audio sau chuẩn hóa dài khoảng {duration_seconds:.2f}s.",
        ]
        if prep_stats.get("trimmed"):
            notes.append(
                f"Đã tự cắt khoảng lặng đầu/cuối khỏi audio tham chiếu (~{float(prep_stats['trimmed_seconds']):.2f}s)."
            )

        return TranscriptionResult(
            text=text,
            language=str(loaded.get("language") or self.asr_language or "vi"),
            model_id=str(loaded.get("model_id") or self.asr_model_id),
            duration_seconds=duration_seconds,
            inference_seconds=elapsed,
            notes=notes,
        )

    def _prepare_reference_audio_for_gwen(self, reference_audio: Path) -> tuple[Path, float, dict[str, float | bool]]:
        raw_audio_np, raw_sr = _inspect_audio_mono_float(reference_audio)
        duration_seconds = float(len(raw_audio_np) / raw_sr)
        activity_ratio = _estimate_activity_ratio(raw_audio_np, raw_sr)
        prep_stats: dict[str, float | bool] = {
            "trimmed": False,
            "trimmed_seconds": 0.0,
            "original_duration_seconds": duration_seconds,
            "activity_ratio": activity_ratio,
            "activity_ratio_after_trim": activity_ratio,
        }
        if duration_seconds < 2.0:
            raise TTSError("Audio tham chiếu quá ngắn cho Gwen-TTS. Hãy dùng đoạn nói rõ dài tối thiểu khoảng 2 giây, tốt nhất 3-10 giây.")
        if duration_seconds > 18.0:
            raise TTSError("Audio tham chiếu quá dài cho Gwen-TTS. Hãy cắt còn khoảng 3-10 giây để clone giọng ổn định hơn.")
        if activity_ratio < 0.08:
            raise TTSError(
                "Audio tham chiếu có quá nhiều khoảng lặng cho Gwen-TTS. "
                "Hãy cắt còn đoạn nói liên tục, rõ tiếng hơn."
            )

        if reference_audio.suffix.lower() == ".wav":
            return reference_audio, duration_seconds, prep_stats

        target_sr = 24000
        audio_np, _ = _load_audio_mono_float(reference_audio, target_sr=target_sr)
        prepared_path = self.reference_dir / f"{reference_audio.stem}-gwen-24k.wav"
        sf.write(prepared_path, audio_np, target_sr, subtype="PCM_16")
        prep_stats["converted"] = True
        return prepared_path, duration_seconds, prep_stats

    def _gwen_card(self) -> EngineCard:
        module_ok, import_warning = self._probe_gwen_import()
        runtime_warning = self._gwen_runtime_stack_warning() if module_ok else None
        if module_ok and not runtime_warning:
            summary = "Sẵn sàng dùng Gwen-TTS 0.6B cho voice cloning tiếng Việt."
            warning = None
        elif runtime_warning:
            summary = "Gwen-TTS chưa sẵn sàng với runtime hiện tại."
            warning = runtime_warning
        elif import_warning == "Chưa cài gói `qwen_tts`.":
            summary = import_warning
            warning = (
                "Engine Gwen-TTS chưa khả dụng trong môi trường này. "
                "Cài `qwen-tts` rồi khởi động lại ứng dụng."
            )
        else:
            summary = "Gwen-TTS cài chưa đủ dependency."
            warning = import_warning

        return EngineCard(
            id="gwen",
            label="Gwen-TTS",
            headline="Model chủ chốt cho tiếng Việt, fine-tune từ Qwen3-TTS-0.6B để clone giọng tự nhiên hơn.",
            description="Adapter này dùng `qwen_tts.Qwen3TTSModel.from_pretrained(...)` với repo Gwen-TTS trên Hugging Face và dựng reusable clone prompt từ audio tham chiếu.",
            recommended_for="Dùng mặc định khi cần chất lượng clone giọng Việt cao, có GPU CUDA, và có transcript tham chiếu chính xác.",
            output_quality="WAV theo sample rate đầu ra của Qwen3/Gwen-TTS, ưu tiên chất lượng hơn tốc độ.",
            reference_hint="Audio tham chiếu 3-10 giây, một người nói, ít nhạc nền. Bắt buộc nhập đúng transcript của audio mẫu.",
            supports_reference_text=True,
            ready=bool(module_ok and not runtime_warning),
            summary=summary,
            warning=warning,
            metadata={
                "model": self.gwen_model_id,
                "dtype": self.gwen_dtype,
                "attn_implementation": self.gwen_attn_implementation,
                "model_selection": self._gwen_model_selection(),
            },
        )

    def _load_gwen(self, model_spec: dict[str, Any]) -> Any:
        with self._locks["gwen"]:
            cache_key = model_spec["cache_key"]
            if cache_key in self._loaded_models:
                return self._loaded_models[cache_key]

            try:
                import torch
                from qwen_tts import Qwen3TTSModel
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(_format_gwen_import_error(exc)) from exc

            if not torch.cuda.is_available():
                raise TTSError("Gwen-TTS trong project này cần GPU CUDA để chạy local.")

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype_name = str(model_spec["dtype"]).strip().lower()
            if dtype_name not in dtype_map:
                raise TTSError(f"GWEN_DTYPE='{model_spec['dtype']}' không hợp lệ. Dùng `bfloat16`, `float16` hoặc `float32`.")

            runtime_attn = model_spec["attn_implementation"]
            try:
                instance = Qwen3TTSModel.from_pretrained(
                    model_spec["model_id"],
                    device_map="cuda:0",
                    dtype=dtype_map[dtype_name],
                    attn_implementation=runtime_attn,
                )
            except Exception as exc:  # pragma: no cover - depends on external install
                if runtime_attn == "flash_attention_2":
                    try:
                        instance = Qwen3TTSModel.from_pretrained(
                            model_spec["model_id"],
                            device_map="cuda:0",
                            dtype=dtype_map[dtype_name],
                            attn_implementation="sdpa",
                        )
                        runtime_attn = "sdpa"
                    except Exception:
                        raise TTSError(f"Khởi tạo Gwen-TTS thất bại: {_format_gwen_runtime_error(exc)}") from exc
                else:
                    raise TTSError(f"Khởi tạo Gwen-TTS thất bại: {_format_gwen_runtime_error(exc)}") from exc

            try:
                setattr(instance, "_gwen_runtime_attn_implementation", runtime_attn)
            except Exception:
                pass

            self._loaded_models[cache_key] = instance
            return instance

    def _synthesize_with_gwen(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        model_spec: dict[str, Any],
        output_path: Path,
        speed: float,
        gwen_generation_config: dict[str, Any] | None = None,
        pronunciation_overrides: list[tuple[str, str]] | None = None,
    ) -> SynthesisResult:
        gwen = self._load_gwen(model_spec)
        started = time.perf_counter()
        prepared_reference_audio, ref_duration, ref_prep_stats = self._prepare_reference_audio_for_gwen(reference_audio)
        normalized_generation_config = _normalize_gwen_generation_config(
            {
                **(gwen_generation_config or {}),
                "speed": gwen_generation_config.get("speed", speed) if gwen_generation_config else speed,
            }
        )
        generation_kwargs = _build_gwen_generation_kwargs(normalized_generation_config)
        transformed_text, applied_pronunciations = _apply_pronunciation_overrides(text, pronunciation_overrides)
        normalized_text, text_normalization_notes = _normalize_gwen_text(transformed_text)
        if len(normalized_text) <= 320:
            chunks = [normalized_text]
        else:
            chunks = _split_text_for_tts(normalized_text, max_chars=320)
        if not chunks:
            raise TTSError("Văn bản đầu vào rỗng sau khi chuẩn hóa cho Gwen-TTS.")

        runtime_notes: list[str] = []
        generate_call_kwargs = dict(generation_kwargs)
        default_call_plan: list[str] = ["direct_no_language", "direct_with_language"]
        if hasattr(gwen, "create_voice_clone_prompt"):
            default_call_plan.append("voice_clone_prompt")

        preferred_call_mode: str | None = None
        chunk_waves: list[np.ndarray] = []
        sample_rate: Any = None

        for chunk in chunks:
            call_plan = [preferred_call_mode] if preferred_call_mode else []
            call_plan.extend(mode for mode in default_call_plan if mode and mode not in call_plan)
            last_error: Exception | None = None
            call_index = 0
            raw_waves: list[Any] = []

            while call_index < len(call_plan):
                call_mode = call_plan[call_index]
                try:
                    if call_mode == "direct_no_language":
                        wavs, sample_rate = gwen.generate_voice_clone(
                            text=chunk,
                            ref_audio=str(prepared_reference_audio),
                            ref_text=reference_text,
                            **generate_call_kwargs,
                        )
                        if "upstream_direct_flow" not in runtime_notes:
                            runtime_notes.append("upstream_direct_flow")
                        preferred_call_mode = call_mode
                    elif call_mode == "direct_with_language":
                        wavs, sample_rate = gwen.generate_voice_clone(
                            text=chunk,
                            language="Vietnamese",
                            ref_audio=str(prepared_reference_audio),
                            ref_text=reference_text,
                            **generate_call_kwargs,
                        )
                        if "language_fallback" not in runtime_notes:
                            runtime_notes.append("language_fallback")
                        preferred_call_mode = call_mode
                    else:
                        voice_clone_prompt = gwen.create_voice_clone_prompt(
                            ref_audio=str(prepared_reference_audio),
                            ref_text=reference_text,
                            x_vector_only_mode=False,
                        )
                        wavs, sample_rate = gwen.generate_voice_clone(
                            text=chunk,
                            language="Vietnamese",
                            voice_clone_prompt=voice_clone_prompt,
                            **generate_call_kwargs,
                        )
                        if "prompt_fallback" not in runtime_notes:
                            runtime_notes.append("prompt_fallback")
                        preferred_call_mode = call_mode

                    raw_waves = list(wavs) if isinstance(wavs, (list, tuple)) else [wavs]
                    break
                except TypeError as exc:  # pragma: no cover - depends on external install
                    message = str(exc or "")
                    lowered_message = message.lower()
                    last_error = exc
                    if "unexpected keyword argument" in message and "speed" in message and "speed" in generate_call_kwargs:
                        generate_call_kwargs.pop("speed", None)
                        if "speed_not_supported" not in runtime_notes and abs(float(normalized_generation_config["speed"]) - 1.0) > 1e-6:
                            runtime_notes.append("speed_not_supported")
                        continue
                    if call_mode == "direct_no_language" and "language" in lowered_message:
                        call_index += 1
                        continue
                    if call_mode == "direct_with_language" and "language" in lowered_message:
                        call_index += 1
                        continue
                    if call_mode in {"direct_no_language", "direct_with_language"} and (
                        "ref_audio" in lowered_message or "ref_text" in lowered_message
                    ):
                        call_index = max(call_index + 1, len(call_plan) - 1)
                        continue
                    raise TTSError(f"Gwen-TTS sinh audio thất bại: {_format_gwen_runtime_error(exc)}") from exc
                except Exception as exc:  # pragma: no cover - depends on external install
                    last_error = exc
                    raise TTSError(f"Gwen-TTS sinh audio thất bại: {_format_gwen_runtime_error(exc)}") from exc
            else:
                raise TTSError(
                    f"Gwen-TTS sinh audio thất bại: {_format_gwen_runtime_error(last_error or RuntimeError('unknown error'))}"
                )

            chunk_waves.extend(_to_numpy_audio(wave) for wave in raw_waves)

        if not chunk_waves or any(wave.size == 0 for wave in chunk_waves):
            raise TTSError("Gwen-TTS trả về audio rỗng. Hãy thử rút ngắn văn bản hoặc đổi audio tham chiếu.")

        sample_rate_int = int(sample_rate or 24000)
        combined = _crossfade_join(chunk_waves, sample_rate=sample_rate_int)
        sf.write(output_path, combined, sample_rate_int)

        elapsed = time.perf_counter() - started
        runtime_attn = getattr(gwen, "_gwen_runtime_attn_implementation", model_spec["attn_implementation"])
        notes = [
            f"Audio tham chiếu Gwen dài khoảng {ref_duration:.2f}s.",
            "Gwen-TTS dùng transcript tham chiếu để clone giọng trực tiếp theo flow inference của upstream.",
        ]
        if "upstream_direct_flow" in runtime_notes:
            notes.insert(0, "Đã gọi Gwen theo flow inference upstream: truyền `ref_audio/ref_text` trực tiếp, không ép `language`.")
        elif "language_fallback" in runtime_notes:
            notes.insert(0, "Runtime Gwen hiện tại cần `language`; app đã fallback sang gọi với `language=\"Vietnamese\"` cùng `ref_audio/ref_text`.")
        elif "prompt_fallback" in runtime_notes:
            notes.insert(0, "Runtime Gwen hiện tại không nhận `ref_audio/ref_text` trực tiếp; app đã fallback sang `voice_clone_prompt`.")
        if "speed_not_supported" in runtime_notes:
            notes.append("Runtime `qwen-tts` hiện tại không nhận tham số `speed`; Gwen chạy ở tốc độ mặc định của model.")
        if ref_prep_stats.get("converted"):
            notes.append("Reference audio của Gwen đã được convert sang WAV mono 24kHz vì file gốc không phải WAV.")
        if applied_pronunciations:
            pronunciation_preview = "; ".join(
                f"{source} -> {target}"
                for source, target, _count in applied_pronunciations[:3]
            )
            if len(applied_pronunciations) > 3:
                pronunciation_preview += "; ..."
            notes.insert(
                0,
                f"Đã áp dụng {len(applied_pronunciations)} quy tắc phát âm trước khi synthesize: {pronunciation_preview}",
            )
        elif pronunciation_overrides:
            notes.insert(0, "Có quy tắc phát âm được khai báo nhưng không khớp với văn bản đầu vào nên không áp dụng.")
        if text_normalization_notes:
            notes.insert(0, " ".join(text_normalization_notes))
        generation_summary = _summarize_gwen_generation_changes(normalized_generation_config)
        if generation_summary:
            notes.insert(0, f"Gwen advanced settings: {generation_summary}.")
        if ref_prep_stats.get("trimmed"):
            notes.append(
                f"Đã tự cắt khoảng lặng đầu/cuối khỏi audio tham chiếu (~{float(ref_prep_stats['trimmed_seconds']):.2f}s)."
            )
        if float(ref_prep_stats.get("activity_ratio_after_trim", 0.0)) < 0.28:
            notes.append("Audio tham chiếu có mật độ giọng nói hơi thưa; Gwen vẫn chạy được nhưng độ bám giọng có thể giảm.")
        if runtime_attn != model_spec["attn_implementation"]:
            notes.append(f"Gwen-TTS đã fallback attention từ `{model_spec['attn_implementation']}` sang `{runtime_attn}` để load ổn định hơn.")
        if re.search(r"\d|[%$@]|[A-Z]{2,}", text) and not applied_pronunciations:
            notes.append(
                "Text đầu vào có số, ký hiệu hoặc viết tắt; nếu phát âm chưa đẹp, hãy thêm quy tắc ở mục `Cài Đặt Phát Âm` để bám cách đọc mong muốn."
            )
        if len(chunks) > 1:
            notes.append(f"Văn bản được tách thành {len(chunks)} chunk và gọi Gwen tuần tự để giữ nhịp đọc ổn định hơn.")
        if model_spec["key"] != "default":
            notes.insert(0, f"Đã dùng model Gwen tùy chọn: {model_spec['label']}.")

        return SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key=model_spec["key"],
            model_label=model_spec["label"],
            output_path=output_path,
            sample_rate=sample_rate_int,
            duration_seconds=len(combined) / float(sample_rate_int),
            inference_seconds=elapsed,
            chunk_count=len(chunks),
            reference_text_used=True,
            seed=None,
            notes=notes,
        )

    @staticmethod
    def _torch_cuda_available() -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _torch_version_label() -> str | None:
        if importlib.util.find_spec("torch") is None:
            return None
        try:
            import torch

            return str(getattr(torch, "__version__", "") or "").strip() or None
        except Exception:
            return None

    def _gwen_runtime_stack_warning(self) -> str | None:
        version_label = self._torch_version_label()
        if not version_label:
            return "Gwen-TTS cần PyTorch và GPU CUDA để chạy local."

        if not self._torch_cuda_available():
            return "Gwen-TTS cần GPU CUDA để chạy local."

        return None

