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


def _normalize_gwen_text(text: str) -> tuple[str, list[str]]:
    normalized = unicodedata.normalize("NFC", text or "")
    notes: list[str] = []

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
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;!?])([^\s])", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if date_hits:
        notes.append("Đã chuẩn hóa ngày tháng năm viết bằng số sang cách đọc tiếng Việt tự nhiên hơn.")
    if abbreviation_hits:
        notes.append("Đã chuẩn hóa chữ viết tắt tiếng Việt/Latin sang cách đọc tự nhiên hơn.")
    if number_hits:
        notes.append("Đã chuẩn hóa số và đơn vị sang dạng đọc tiếng Việt trước khi synthesize.")
    return normalized, notes


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


def _parse_enabled_engines(raw: str) -> list[str]:
    if not raw:
        return ["gwen", "vieneu", "f5"]

    engines: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[\s,;|]+", raw):
        normalized = _normalize_engine_id(item, default="")
        if normalized not in {"gwen", "vieneu", "f5"} or normalized in seen:
            continue
        engines.append(normalized)
        seen.add(normalized)
    return engines or ["gwen", "vieneu", "f5"]


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
        "transformers_stream_generator": "transformers-stream-generator",
        "unidecode": "unidecode",
        "x_transformers": "x-transformers",
    }.get((module_name or "").strip(), (module_name or "").strip())


def _is_torch_audio_abi_mismatch_message(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(message or "")).strip().lower()
    return any(
        marker in normalized
        for marker in (
            "undefined symbol: torch_library_impl",
            "undefined symbol: torch_list_push_back",
            "_torchaudio.abi3.so",
            "libtorchaudio.so",
            "pytorch version",
            "not compatible with",
        )
    )


def _format_f5_import_error(exc: Exception) -> str:
    if _is_torch_audio_abi_mismatch_message(str(exc)):
        return (
            "F5-TTS không import được vì stack `torch`/`torchaudio` trong runtime đang lệch ABI. "
            "Nếu đang dùng notebook Colab của repo này, hãy rerun cell cài dependencies bản mới "
            "để dọn sạch package cũ và cài lại đúng cặp `torch==2.11.0` với `torchaudio==2.11.0`."
        )
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        missing_module = exc.name.strip()
        package_name = _map_import_name_to_package(missing_module)
        return (
            f"F5-TTS đã được cài nhưng thiếu dependency `{missing_module}`. "
            f"Hãy chạy `python -m pip install -U f5-tts {package_name}` rồi khởi động lại ứng dụng. "
            "Nếu đang dùng notebook Colab của repo này, hãy dùng cell cài dependencies bản mới "
            "vì bản cũ cài `f5-tts` với `--no-deps`."
        )
    return f"Không thể import F5-TTS: {exc}"


def _format_f5_runtime_error(exc: Exception) -> str:
    normalized = re.sub(r"\s+", " ", str(exc or "")).strip().lower()
    if _is_torch_audio_abi_mismatch_message(normalized) or any(
        marker in normalized
        for marker in (
            "could not load libtorchaudio codec",
            "ffmpeg is not properly installed",
        )
    ):
        return (
            "Không đọc được codec audio của torchaudio/FFmpeg trong runtime hiện tại. "
            "Notebook của repo này đã chuyển mặc định sang Gwen-TTS; "
            "hãy đổi engine sang `Gwen-TTS` hoặc `VieNeu-TTS`, hoặc rerun cell cài dependencies "
            "để đồng bộ torch/torchaudio và ffmpeg trước khi dùng F5."
        )
    return str(exc)


def _format_vieneu_import_error(exc: Exception) -> str:
    if _is_torch_audio_abi_mismatch_message(str(exc)):
        return (
            "VieNeu-TTS không import được vì stack `torch`/`torchaudio` trong runtime đang lệch ABI. "
            "Nếu đang dùng notebook Colab của repo này, hãy rerun cell cài dependencies bản mới "
            "để dọn sạch package cũ và cài lại đúng cặp `torch==2.11.0` với `torchaudio==2.11.0`."
        )
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
        missing_module = exc.name.strip()
        package_name = _map_import_name_to_package(missing_module)
        return (
            f"VieNeu-TTS đã được cài nhưng thiếu dependency `{missing_module}`. "
            f"Hãy chạy `python -m pip install -U vieneu==2.4.3 {package_name}` rồi khởi động lại ứng dụng. "
            "Nếu đang dùng notebook Colab của repo này, hãy rerun cell cài dependencies mới "
            "vì `pip install vieneu[gpu]` không tự resolve đủ wheel GPU upstream."
        )
    return f"Không thể import VieNeu-TTS: {exc}"


def _format_vieneu_runtime_error(exc: Exception) -> str:
    normalized = re.sub(r"\s+", " ", str(exc or "")).strip()
    lowered = normalized.lower()
    if _is_torch_audio_abi_mismatch_message(lowered) or any(
        marker in lowered
        for marker in (
            "requires pytorch",
            "install torch via: pip install vieneu[gpu]",
            "torch >= 2.11.0",
            "skipping import of cpp extensions",
            "remote mode",
            "neucodec-onnx-decoder-int8",
        )
    ):
        return (
            "VieNeu Standard local không khởi tạo được với stack PyTorch hiện tại. "
            "Rerun cell cài dependencies mới của notebook để cài lại `torch`/`torchaudio` theo CUDA 12.8 "
            "và `vieneu[gpu]`, rồi khởi động lại webapp. "
            "Nếu runtime không có GPU NVIDIA, hãy đổi `VIENEU_MODE=turbo`."
        )
    return normalized


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


def _normalize_engine_id(value: str, default: str = "gwen") -> str:
    normalized = (value or "").strip().lower()
    if normalized == "vira":
        return "vieneu"
    return normalized or default


def _vieneu_mode_requires_reference_text(mode_name: str) -> bool:
    normalized = (mode_name or "").strip().lower()
    return normalized not in {"turbo", "turbo_gpu"}


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


def _split_text_by_pause(text: str, max_chars: int = 60) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    segments = re.findall(r"[^,;:]+(?:[,;:]|$)", normalized)
    if len(segments) <= 1:
        return [normalized]

    chunks: list[str] = []
    current = ""
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        candidate = segment if not current else f"{current} {segment}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = segment

    if current:
        chunks.append(current)
    return chunks or [normalized]


def _split_failed_vira_chunk(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    strategies = [
        lambda value: _split_text_for_tts(value, max_chars=80),
        lambda value: _split_text_for_tts(value, max_chars=60),
        lambda value: _split_text_by_pause(value, max_chars=56),
        lambda value: _split_text_by_words(value, max_words=8, max_chars=52),
        lambda value: _split_text_by_words(value, max_words=6, max_chars=44),
        lambda value: _split_text_by_words(value, max_words=4, max_chars=34),
    ]
    seen: set[tuple[str, ...]] = set()

    for strategy in strategies:
        parts = [part.strip() for part in strategy(normalized) if part.strip()]
        signature = tuple(parts)
        if len(parts) <= 1 or signature in seen:
            continue
        seen.add(signature)
        return parts

    return [normalized]


def _fallback_cleanup_vira_text(
    text: str,
    *,
    aggressive: bool = False,
    ensure_terminal: bool = True,
) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""

    replacements = [
        ("...", ", "),
        ("…", ", "),
        ("—", "-"),
        ("–", "-"),
        ("\u00a0", " "),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)

    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(?=[^\s\d])", r"\1 ", cleaned)

    if aggressive:
        cleaned = re.sub(r"[\"“”`*_#<>\[\]{}()]+", " ", cleaned)
        cleaned = re.sub(r"\s*[|/]\s*", ", ", cleaned)
        cleaned = re.sub(r"\s*[-:;]\s*", ", ", cleaned)
        cleaned = re.sub(r",\s*,+", ", ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if ensure_terminal and cleaned and cleaned[-1] not in ".!?,":
        cleaned += "."
    return cleaned


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


class TTSStudioService:
    """Flask-facing service that orchestrates Gwen-TTS, VieNeu-TTS, and F5-TTS engines."""

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

        self.enabled_engines = _parse_enabled_engines(os.getenv("TTS_ENABLED_ENGINES", ""))
        self.default_engine = _normalize_engine_id(os.getenv("TTS_DEFAULT_ENGINE", "gwen"))
        if self.default_engine not in self.enabled_engines:
            self.default_engine = self.enabled_engines[0]
        self.gwen_model_id = os.getenv("GWEN_MODEL_ID", "g-group-ai-lab/gwen-tts-0.6B").strip() or "g-group-ai-lab/gwen-tts-0.6B"
        self.gwen_model_choices = os.getenv("GWEN_MODEL_CHOICES", "").strip()
        self.gwen_dtype = (os.getenv("GWEN_DTYPE", "bfloat16") or "bfloat16").strip().lower() or "bfloat16"
        self.gwen_attn_implementation = (os.getenv("GWEN_ATTN_IMPLEMENTATION", "flash_attention_2") or "flash_attention_2").strip() or "flash_attention_2"
        self.vieneu_mode = (os.getenv("VIENEU_MODE", "standard") or "standard").strip().lower() or "standard"
        self.vieneu_mode_choices = os.getenv("VIENEU_MODE_CHOICES", "").strip()

        self.f5_model_name = os.getenv("F5_MODEL_NAME", "F5TTS_v1_Base")
        self.f5_ckpt_file = os.getenv("F5_CKPT_FILE", "").strip()
        self.f5_vocab_file = os.getenv("F5_VOCAB_FILE", "").strip()
        self.f5_vocoder_local_path = os.getenv("F5_VOCODER_LOCAL_PATH", "").strip() or None
        self.f5_model_choices = os.getenv("F5_MODEL_CHOICES", "").strip()
        self._locks = {
            "gwen": Lock(),
            "f5": Lock(),
            "vieneu": Lock(),
        }
        self._loaded_models: dict[str, Any] = {}
        self._gwen_import_probe: tuple[bool, str | None] | None = None
        self._f5_import_probe: tuple[bool, str | None] | None = None
        self._vieneu_import_probe: tuple[bool, str | None] | None = None
        self._gwen_preset_voices_cache: list[PresetVoice] | None = None

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(path)

    @staticmethod
    def _make_gwen_model_key(model_id: str) -> str:
        return f"model::{model_id.strip()}"

    @staticmethod
    def _make_f5_model_key(model_name: str) -> str:
        return f"name::{model_name.strip()}"

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

    def get_preset_voices(self, engine_id: str) -> list[PresetVoice]:
        engine = _normalize_engine_id(engine_id)
        if engine != "gwen":
            return []
        return self._load_gwen_preset_voices()

    def get_preset_voice(self, engine_id: str, preset_voice_id: str) -> PresetVoice:
        engine = _normalize_engine_id(engine_id)
        if engine != "gwen":
            raise TTSError("Preset voice hiện chỉ hỗ trợ cho Gwen-TTS.")

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

    def _probe_f5_import(self) -> tuple[bool, str | None]:
        if self._f5_import_probe is not None:
            return self._f5_import_probe

        if importlib.util.find_spec("f5_tts") is None:
            self._f5_import_probe = (False, "Chưa cài gói `f5_tts`.")
            return self._f5_import_probe

        try:
            importlib.import_module("f5_tts.api")
        except Exception as exc:
            self._f5_import_probe = (False, _format_f5_import_error(exc))
            return self._f5_import_probe

        self._f5_import_probe = (True, None)
        return self._f5_import_probe

    def _probe_vieneu_import(self) -> tuple[bool, str | None]:
        if self._vieneu_import_probe is not None:
            return self._vieneu_import_probe

        if importlib.util.find_spec("vieneu") is None:
            self._vieneu_import_probe = (False, "Chưa cài gói `vieneu`.")
            return self._vieneu_import_probe

        try:
            importlib.import_module("vieneu")
        except Exception as exc:
            self._vieneu_import_probe = (False, _format_vieneu_import_error(exc))
            return self._vieneu_import_probe

        self._vieneu_import_probe = (True, None)
        return self._vieneu_import_probe

    @staticmethod
    def _make_vieneu_mode_key(mode_name: str) -> str:
        return f"mode::{(mode_name or '').strip().lower()}"

    @staticmethod
    def _normalize_vieneu_mode(mode_name: str) -> str:
        normalized = (mode_name or "").strip().lower()
        if normalized not in {"turbo", "turbo_gpu", "fast", "standard"}:
            raise TTSError("Mode VieNeu không hợp lệ.")
        return normalized

    def _f5_model_selection(self) -> dict[str, Any]:
        options = [
            {
                "key": "default",
                "label": f"Mặc định: {self.f5_model_name}",
                "value": self.f5_model_name,
                "mode": "default",
            }
        ]
        seen_keys = {"default"}
        for label, value in _parse_named_choices(self.f5_model_choices):
            key = self._make_f5_model_key(value)
            if key in seen_keys:
                continue
            options.append(
                {
                    "key": key,
                    "label": label,
                    "value": value,
                    "mode": "name",
                }
            )
            seen_keys.add(key)

        return {
            "default_key": "default",
            "allow_custom": True,
            "custom_placeholder": "Ví dụ: F5TTS_v1_Base hoặc model name upstream khác",
            "help_primary": "F5 nhận model name theo upstream; chọn custom nếu muốn thử model khác mà không sửa biến môi trường.",
            "help_secondary": "Checkpoint, vocab và vocoder vẫn lấy từ cấu hình hiện tại nếu bạn đã set biến môi trường.",
            "options": options,
        }

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

    def _vieneu_mode_selection(self) -> dict[str, Any]:
        default_requires_reference_text = _vieneu_mode_requires_reference_text(self.vieneu_mode)
        options = [
            {
                "key": "default",
                "label": f"Mặc định: VieNeu {self.vieneu_mode}",
                "value": self.vieneu_mode,
                "mode": "default",
                "requires_reference_text": default_requires_reference_text,
            }
        ]
        seen_keys = {"default"}

        for label, value in _parse_named_choices(self.vieneu_mode_choices):
            mode_name = self._normalize_vieneu_mode(value)
            key = self._make_vieneu_mode_key(mode_name)
            if key in seen_keys:
                continue
            options.append(
                {
                    "key": key,
                    "label": label,
                    "value": mode_name,
                    "mode": "named",
                    "requires_reference_text": _vieneu_mode_requires_reference_text(mode_name),
                }
            )
            seen_keys.add(key)

        return {
            "default_key": "default",
            "allow_custom": True,
            "default_value": self.vieneu_mode,
            "custom_placeholder": "Ví dụ: standard, fast, turbo hoặc turbo_gpu",
            "help_primary": (
                "VieNeu đang mặc định `standard` để ưu tiên chất lượng và độ bám giọng. "
                "Hãy nhập transcript đúng với audio tham chiếu."
                if default_requires_reference_text
                else "VieNeu đang mặc định `turbo` để ưu tiên ổn định và clone giọng nhanh hơn."
            ),
            "help_secondary": (
                "Mode `standard` và `fast` bắt buộc transcript tham chiếu; "
                "`turbo` và `turbo_gpu` nhẹ hơn nhưng chất lượng thường thấp hơn."
            ),
            "options": options,
        }

    def get_model_selection(self, engine_id: str) -> dict[str, Any]:
        engine = _normalize_engine_id(engine_id)
        if engine == "gwen":
            return self._gwen_model_selection()
        if engine == "f5":
            return self._f5_model_selection()
        if engine == "vieneu":
            return self._vieneu_mode_selection()
        raise TTSError(f"Engine '{engine_id}' không tồn tại.")

    def resolve_model_spec(
        self,
        engine_id: str,
        *,
        model_key: str = "default",
        custom_model: str = "",
    ) -> dict[str, Any]:
        engine = _normalize_engine_id(engine_id)
        key = (model_key or "default").strip() or "default"
        custom = (custom_model or "").strip()

        if engine == "gwen":
            if key == "__custom__":
                if not custom:
                    raise TTSError("Hãy nhập model Gwen tùy chỉnh trước khi generate.")
                model_id = custom
            elif key == "default":
                model_id = self.gwen_model_id
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

        if engine == "f5":
            if key == "__custom__":
                if not custom:
                    raise TTSError("Hãy nhập model F5 tùy chỉnh trước khi generate.")
                return {
                    "key": self._make_f5_model_key(custom),
                    "label": custom,
                    "mode": "name",
                    "model_name": custom,
                    "cache_key": f"f5::{custom}::{self.f5_ckpt_file}::{self.f5_vocab_file}::{self.f5_vocoder_local_path or ''}",
                }
            if key == "default":
                return {
                    "key": "default",
                    "label": self.f5_model_name,
                    "mode": "default",
                    "model_name": self.f5_model_name,
                    "cache_key": f"f5::default::{self.f5_model_name}::{self.f5_ckpt_file}::{self.f5_vocab_file}::{self.f5_vocoder_local_path or ''}",
                }
            if key.startswith("name::"):
                model_name = key.split("::", 1)[1].strip()
                if not model_name:
                    raise TTSError("Model F5 không hợp lệ.")
                return {
                    "key": key,
                    "label": model_name,
                    "mode": "name",
                    "model_name": model_name,
                    "cache_key": f"f5::{model_name}::{self.f5_ckpt_file}::{self.f5_vocab_file}::{self.f5_vocoder_local_path or ''}",
                }
            raise TTSError("Lựa chọn model F5 không hợp lệ.")

        if engine == "vieneu":
            if key == "__custom__":
                if not custom:
                    raise TTSError("Hãy nhập mode VieNeu tùy chỉnh trước khi generate.")
                mode_name = self._normalize_vieneu_mode(custom)
            elif key == "default":
                return {
                    "key": "default",
                    "label": f"VieNeu {self.vieneu_mode}",
                    "mode": "default",
                    "mode_name": self.vieneu_mode,
                    "cache_key": f"vieneu::default::{self.vieneu_mode}",
                }
            elif key.startswith("mode::"):
                mode_name = self._normalize_vieneu_mode(key.split("::", 1)[1])
            else:
                raise TTSError("Lựa chọn mode VieNeu không hợp lệ.")
            return {
                "key": self._make_vieneu_mode_key(mode_name),
                "label": f"VieNeu {mode_name}",
                "mode": "name",
                "mode_name": mode_name,
                "cache_key": f"vieneu::{mode_name}",
            }

        raise TTSError(f"Engine '{engine_id}' không tồn tại.")

    def summary(self) -> dict[str, Any]:
        cards = self.get_engine_cards()
        ready_count = sum(1 for card in cards if card.ready)
        device_label = "GPU" if self._torch_cuda_available() else "CPU"
        default_card = next((card for card in cards if card.id == _normalize_engine_id(self.default_engine)), cards[0])
        return {
            "engine_count": len(cards),
            "ready_count": ready_count,
            "device_label": device_label,
            "default_engine": default_card.label,
        }

    def get_engine_cards(self) -> list[EngineCard]:
        cards_by_id = {
            "gwen": self._gwen_card(),
            "vieneu": self._vieneu_card(),
            "f5": self._f5_card(),
        }
        return [cards_by_id[engine_id] for engine_id in self.enabled_engines]

    def get_engine_card(self, engine_id: str) -> EngineCard:
        engine = _normalize_engine_id(engine_id)
        for card in self.get_engine_cards():
            if card.id == engine:
                return card
        raise TTSError(f"Engine '{engine_id}' không tồn tại.")

    def synthesize(
        self,
        *,
        engine_id: str,
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
        text = re.sub(r"\s+", " ", (text or "").strip())
        reference_text = re.sub(r"\s+", " ", (reference_text or "").strip())

        if not text:
            raise TTSError("Thiếu nội dung văn bản cần chuyển giọng nói.")
        if len(text) > TEXT_INPUT_LIMIT:
            raise TTSError(f"Văn bản quá dài. Giới hạn hiện tại là {TEXT_INPUT_LIMIT} ký tự mỗi lượt.")
        if not reference_audio.exists():
            raise TTSError("Không tìm thấy file audio tham chiếu.")

        engine = self.get_engine_card(engine_id)
        if not engine.ready:
            raise TTSError(engine.warning or engine.summary)
        model_spec = self.resolve_model_spec(
            engine.id,
            model_key=model_key,
            custom_model=custom_model,
        )
        if engine.id == "gwen" and not reference_text:
            raise TTSError(
                "Gwen-TTS cần transcript tham chiếu. Hãy nhập đúng câu đang có trong audio tham chiếu trước khi generate."
            )
        if engine.id == "vieneu" and _vieneu_mode_requires_reference_text(model_spec["mode_name"]) and not reference_text:
            raise TTSError(
                f"Mode VieNeu `{model_spec['mode_name']}` cần transcript tham chiếu. "
                "Hãy nhập đúng câu đang có trong audio tham chiếu trước khi generate."
            )

        output_path = self.output_dir / f"{int(time.time())}-{uuid.uuid4().hex[:10]}-{engine.id}.wav"
        if engine.id == "gwen":
            return self._synthesize_with_gwen(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                model_spec=model_spec,
                output_path=output_path,
                speed=speed,
                gwen_generation_config=gwen_generation_config,
                pronunciation_overrides=pronunciation_overrides,
            )
        if engine.id == "f5":
            return self._synthesize_with_f5(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                speed=speed,
                remove_silence=remove_silence,
                seed=seed,
                model_spec=model_spec,
                output_path=output_path,
            )
        if engine.id == "vieneu":
            return self._synthesize_with_vieneu(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                model_spec=model_spec,
                output_path=output_path,
            )
        raise TTSError(f"Engine '{engine_id}' không được hỗ trợ.")

    def save_reference_file(self, filename: str, payload: bytes) -> Path:
        extension = Path(filename).suffix.lower() or ".wav"
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", Path(filename).stem).strip("-") or "reference"
        saved = self.reference_dir / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{safe_name}{extension}"
        saved.write_bytes(payload)
        return saved

    def _prepare_reference_audio_for_vieneu(self, reference_audio: Path) -> tuple[Path, float, dict[str, float | bool]]:
        target_sr = 24000
        audio_np, sr = _load_audio_mono_float(reference_audio, target_sr=target_sr)
        audio_np, prep_stats = _trim_reference_silence(audio_np, sr)
        duration_seconds = float(len(audio_np) / sr)
        activity_after_trim = _estimate_activity_ratio(audio_np, sr)
        if duration_seconds < 1.0:
            raise TTSError("Audio tham chiếu quá ngắn cho VieNeu. Hãy dùng đoạn nói rõ dài tối thiểu khoảng 1 giây, tốt nhất 3-8 giây.")
        if duration_seconds > 15.0:
            raise TTSError("Audio tham chiếu quá dài cho VieNeu. Hãy cắt còn khoảng 3-8 giây để clone giọng ổn định hơn.")
        if activity_after_trim < 0.12:
            raise TTSError(
                "Audio tham chiếu có quá nhiều khoảng lặng cho VieNeu. "
                "Hãy cắt còn 3-8 giây giọng nói liên tục, rõ tiếng hơn."
            )

        prepared_path = self.reference_dir / f"{reference_audio.stem}-vieneu-24k.wav"
        sf.write(prepared_path, audio_np, target_sr)
        prep_stats["activity_ratio_after_trim"] = activity_after_trim
        return prepared_path, duration_seconds, prep_stats

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

    def _prepare_reference_audio_for_f5(self, reference_audio: Path) -> tuple[Path, float, list[str]]:
        target_sr = 24000
        prepared_path = self.reference_dir / f"{reference_audio.stem}-f5-24k.wav"
        suffix = reference_audio.suffix.lower()
        notes: list[str] = []
        compressed_suffixes = {".mp3", ".m4a", ".ogg", ".aac", ".wma", ".webm", ".mp4"}

        if suffix in compressed_suffixes:
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise TTSError(
                    "F5-TTS cần `ffmpeg` để đọc audio tham chiếu dạng MP3/M4A/OGG. "
                    "Hãy cài `ffmpeg` hoặc đổi file tham chiếu sang WAV/FLAC rồi thử lại."
                )
            try:
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-i",
                        str(reference_audio),
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sr),
                        "-c:a",
                        "pcm_s16le",
                        str(prepared_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as exc:
                raise TTSError(
                    "Không thể convert audio tham chiếu sang WAV cho F5-TTS bằng ffmpeg: "
                    f"{_summarize_subprocess_error(exc)}"
                ) from exc
            audio_np, _ = _load_audio_mono_float(prepared_path, target_sr=target_sr)
            sf.write(prepared_path, audio_np, target_sr, subtype="PCM_16")
            notes.append("Đã tự convert audio tham chiếu sang WAV 24kHz trước khi gọi F5-TTS để tránh lỗi codec.")
        else:
            audio_np, _ = _load_audio_mono_float(reference_audio, target_sr=target_sr)
            sf.write(prepared_path, audio_np, target_sr, subtype="PCM_16")
            notes.append("Đã chuẩn hóa audio tham chiếu về WAV mono 24kHz trước khi gọi F5-TTS.")

        duration_seconds = float(len(audio_np) / target_sr)
        if duration_seconds < 1.0:
            raise TTSError("Audio tham chiếu quá ngắn cho F5-TTS. Hãy dùng đoạn nói rõ dài tối thiểu khoảng 1 giây.")
        if duration_seconds > 20.0:
            notes.append("Audio tham chiếu cho F5-TTS hơi dài; tốt nhất nên giữ khoảng 3-12 giây để clone giọng ổn định hơn.")

        return prepared_path, duration_seconds, notes

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

    def _f5_card(self) -> EngineCard:
        module_ok, import_warning = self._probe_f5_import()
        if module_ok:
            summary = "Sẵn sàng dùng zero-shot voice cloning với F5-TTS."
            warning = None
        elif import_warning == "Chưa cài gói `f5_tts`.":
            summary = import_warning
            warning = (
                "Engine F5-TTS chưa khả dụng trong môi trường này. "
                "Cài upstream F5-TTS rồi khởi động lại ứng dụng."
            )
        else:
            summary = "F5-TTS cài chưa đủ dependency."
            warning = import_warning
        return EngineCard(
            id="f5",
            label="F5-TTS",
            headline="Voice cloning linh hoạt, hợp cho voice-over và thử nghiệm đa phong cách.",
            description="Adapter này gọi API `F5TTS().infer(...)` của F5-TTS và tự chia câu dài thành nhiều chunk nhỏ để ổn định hơn.",
            recommended_for="Dùng khi cần clone giọng tự nhiên, giữ nhịp đọc mềm và muốn tinh chỉnh speed / seed.",
            output_quality="Output WAV theo sample rate của model F5 đã nạp.",
            reference_hint="Audio tham chiếu 3-12 giây, nói rõ, ít nhạc nền. Có thể nhập transcript để giữ nội dung tham chiếu chính xác hơn.",
            supports_reference_text=True,
            ready=module_ok,
            summary=summary,
            warning=warning,
            metadata={
                "model": self.f5_model_name,
                "checkpoint": self.f5_ckpt_file or "auto-download theo cấu hình upstream",
                "model_selection": self._f5_model_selection(),
            },
        )

    def _vieneu_card(self) -> EngineCard:
        mode_name = self.vieneu_mode
        requires_reference_text = _vieneu_mode_requires_reference_text(mode_name)
        module_ok, import_warning = self._probe_vieneu_import()
        runtime_warning = self._vieneu_runtime_stack_warning(mode_name) if module_ok else None
        if module_ok and not runtime_warning:
            summary = f"Sẵn sàng dùng VieNeu-TTS {mode_name} cho tiếng Việt 24kHz."
            warning = None
        elif runtime_warning:
            summary = f"VieNeu-TTS {mode_name} chưa sẵn sàng với runtime hiện tại."
            warning = runtime_warning
        elif import_warning == "Chưa cài gói `vieneu`.":
            summary = import_warning
            warning = (
                "Engine VieNeu-TTS chưa khả dụng trong môi trường này. "
                "Cài `vieneu[gpu]` rồi khởi động lại ứng dụng nếu muốn dùng mode standard."
            )
        else:
            summary = "VieNeu-TTS cài chưa đủ dependency."
            warning = import_warning

        return EngineCard(
            id="vieneu",
            label="VieNeu-TTS",
            headline=(
                "Engine tiếng Việt chính của project, ưu tiên chất lượng clone giọng và độ bám lời."
                if requires_reference_text
                else "Engine tiếng Việt ổn định hơn ViRa, clone giọng trực tiếp từ audio tham chiếu."
            ),
            description=(
                f"Adapter này dùng `vieneu.Vieneu(mode='{mode_name}')` để encode giọng tham chiếu "
                "và sinh audio 24kHz, tránh flow LLM speech-token dễ lỗi của ViRa."
            ),
            recommended_for=(
                "Dùng mặc định khi cần chất lượng cao hơn, clone giọng sát hơn và chấp nhận nhập transcript tham chiếu."
                if requires_reference_text
                else "Dùng khi cần tiếng Việt ổn định, ít lỗi và không muốn nhập transcript tham chiếu."
            ),
            output_quality=(
                f"24kHz audio từ VieNeu {mode_name}, ưu tiên chất lượng và độ bám giọng."
                if requires_reference_text
                else f"24kHz audio từ VieNeu {mode_name}, ưu tiên ổn định và tốc độ."
            ),
            reference_hint=(
                "Audio tham chiếu 3-8 giây, một người nói, ít nhạc nền. "
                "Bắt buộc nhập transcript đúng với câu đang có trong audio mẫu."
                if requires_reference_text
                else "Audio tham chiếu 3-8 giây, một người nói, ít nhạc nền. Transcript tham chiếu không bắt buộc."
            ),
            supports_reference_text=requires_reference_text,
            ready=bool(module_ok and not runtime_warning),
            summary=summary,
            warning=warning,
            metadata={
                "mode": mode_name,
                "requires_reference_text": requires_reference_text,
                "model_selection": self._vieneu_mode_selection(),
            },
        )

    def _load_f5(self, model_spec: dict[str, Any]) -> Any:
        with self._locks["f5"]:
            cache_key = model_spec["cache_key"]
            if cache_key in self._loaded_models:
                return self._loaded_models[cache_key]

            try:
                from f5_tts.api import F5TTS
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(_format_f5_import_error(exc)) from exc

            kwargs: dict[str, Any] = {
                "model": model_spec["model_name"],
            }
            if self.f5_ckpt_file:
                kwargs["ckpt_file"] = self.f5_ckpt_file
            if self.f5_vocab_file:
                kwargs["vocab_file"] = self.f5_vocab_file
            if self.f5_vocoder_local_path:
                kwargs["vocoder_local_path"] = self.f5_vocoder_local_path

            try:
                instance = F5TTS(**kwargs)
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Khởi tạo F5-TTS thất bại: {exc}") from exc

            self._loaded_models[cache_key] = instance
            return instance

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

    def _load_vieneu(self, model_spec: dict[str, Any]) -> Any:
        with self._locks["vieneu"]:
            cache_key = model_spec["cache_key"]
            if cache_key in self._loaded_models:
                return self._loaded_models[cache_key]

            try:
                from vieneu import Vieneu
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(_format_vieneu_import_error(exc)) from exc

            try:
                instance = Vieneu(mode=model_spec["mode_name"])
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(f"Khởi tạo VieNeu-TTS thất bại: {_format_vieneu_runtime_error(exc)}") from exc

            self._loaded_models[cache_key] = instance
            return instance

    def _ensure_vira_model_path(self, model_spec: dict[str, Any]) -> Path:
        if model_spec["mode"] == "path":
            model_path = Path(model_spec["model_path"])
            if _nonempty_dir(model_path):
                return model_path
            raise TTSError(
                f"Không tìm thấy model ViRa tại '{model_path}'. "
                "Hãy kiểm tra lại đường dẫn local."
            )

        if model_spec["mode"] == "repo":
            model_path = Path(model_spec["download_path"])
            repo_id = model_spec["repo_id"]
            if _nonempty_dir(model_path):
                return model_path
            if not self.vira_auto_download:
                raise TTSError(
                    f"Model ViRa '{repo_id}' chưa có trong cache local. "
                    "Bật `VIRA_AUTO_DOWNLOAD=1` để app tự tải repo này."
                )
            return self._download_vira_repo(repo_id=repo_id, local_dir=model_path)

        if _nonempty_dir(self.vira_model_path):
            return self.vira_model_path

        if not self.vira_auto_download:
            raise TTSError(
                f"Thiếu model ViRa tại '{self.vira_model_path}'. "
                "Bật `VIRA_AUTO_DOWNLOAD=1` hoặc tải model thủ công."
            )

        return self._download_vira_repo(repo_id=self.vira_model_id, local_dir=self.vira_model_path)

    def _download_vira_repo(self, *, repo_id: str, local_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"Không thể auto-download model ViRa: {exc}") from exc

        local_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return local_dir

    def _synthesize_with_f5(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        speed: float,
        remove_silence: bool,
        seed: int | None,
        model_spec: dict[str, Any],
        output_path: Path,
    ) -> SynthesisResult:
        f5 = self._load_f5(model_spec)
        prepared_reference_audio, ref_duration, prep_notes = self._prepare_reference_audio_for_f5(reference_audio)
        notes: list[str] = list(prep_notes)
        chunks = _split_text_for_tts(text, max_chars=220)
        chunk_waves: list[np.ndarray] = []
        sample_rate = None
        started = time.perf_counter()

        for index, chunk in enumerate(chunks):
            chunk_seed = None if seed is None else seed + index
            try:
                wav, sr, _ = f5.infer(
                    ref_file=str(prepared_reference_audio),
                    ref_text=reference_text,
                    gen_text=chunk,
                    speed=float(speed),
                    remove_silence=False,
                    seed=chunk_seed,
                    show_info=lambda *_args, **_kwargs: None,
                )
            except Exception as exc:  # pragma: no cover - depends on external install
                raise TTSError(
                    f"F5-TTS sinh audio thất bại ở chunk {index + 1}: {_format_f5_runtime_error(exc)}"
                ) from exc

            wave = _to_numpy_audio(wav)
            chunk_waves.append(wave)
            sample_rate = int(sr)

        combined = _crossfade_join(chunk_waves, sample_rate=sample_rate or 24000)
        sf.write(output_path, combined, sample_rate or 24000)

        if remove_silence:
            try:
                from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

                remove_silence_for_generated_wav(str(output_path))
                notes.append("Đã loại bớt khoảng lặng bằng tiện ích upstream của F5-TTS.")
            except Exception:
                notes.append("Không thể remove silence tự động, nên giữ nguyên audio gốc.")

        elapsed = time.perf_counter() - started
        duration = len(combined) / float(sample_rate or 24000)
        notes.append(f"Audio tham chiếu F5 sau chuẩn hóa dài khoảng {ref_duration:.2f}s.")
        if len(chunks) > 1:
            notes.append(f"Văn bản được tách thành {len(chunks)} chunk để giữ inference ổn định hơn.")
        if reference_text:
            notes.append("Đã dùng transcript tham chiếu để neo chất giọng của F5-TTS.")
        if model_spec["key"] != "default":
            notes.insert(0, f"Đã dùng model F5 tùy chọn: {model_spec['label']}.")

        return SynthesisResult(
            engine_id="f5",
            engine_label="F5-TTS",
            model_key=model_spec["key"],
            model_label=model_spec["label"],
            output_path=output_path,
            sample_rate=sample_rate or 24000,
            duration_seconds=duration,
            inference_seconds=elapsed,
            chunk_count=len(chunks),
            reference_text_used=bool(reference_text),
            seed=seed,
            notes=notes,
        )

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
        default_call_plan: list[str] = ["direct_with_language", "direct_no_language"]
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
                    if call_mode == "direct_with_language":
                        wavs, sample_rate = gwen.generate_voice_clone(
                            text=chunk,
                            language="Vietnamese",
                            ref_audio=str(prepared_reference_audio),
                            ref_text=reference_text,
                            **generate_call_kwargs,
                        )
                        if "official_language_flow" not in runtime_notes:
                            runtime_notes.append("official_language_flow")
                        preferred_call_mode = call_mode
                    elif call_mode == "direct_no_language":
                        wavs, sample_rate = gwen.generate_voice_clone(
                            text=chunk,
                            ref_audio=str(prepared_reference_audio),
                            ref_text=reference_text,
                            **generate_call_kwargs,
                        )
                        if "no_language_fallback" not in runtime_notes:
                            runtime_notes.append("no_language_fallback")
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
        if "official_language_flow" in runtime_notes:
            notes.insert(0, "Đã gọi Gwen theo khuyến nghị chính thức: truyền `language=\"Vietnamese\"` cùng `ref_audio/ref_text`.")
        elif "no_language_fallback" in runtime_notes:
            notes.insert(0, "Runtime `qwen-tts` hiện tại không nhận `language`; app đã fallback sang gọi trực tiếp với `ref_audio/ref_text`.")
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

    def _synthesize_with_vieneu(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        model_spec: dict[str, Any],
        output_path: Path,
    ) -> SynthesisResult:
        vieneu = self._load_vieneu(model_spec)
        started = time.perf_counter()
        prepared_reference_audio, ref_duration, ref_prep_stats = self._prepare_reference_audio_for_vieneu(reference_audio)

        try:
            voice = vieneu.encode_reference(str(prepared_reference_audio))
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"VieNeu không encode được audio tham chiếu: {exc}") from exc

        if _numel(voice) == 0:
            raise TTSError("VieNeu encode ra voice embedding rỗng từ audio tham chiếu.")

        requires_reference_text = _vieneu_mode_requires_reference_text(model_spec["mode_name"])
        infer_kwargs: dict[str, Any] = {
            "text": text,
            "max_chars": 220,
            "show_progress": False,
        }
        if not requires_reference_text:
            infer_kwargs["voice"] = voice
        else:
            if not reference_text:
                raise TTSError(
                    "Mode VieNeu này cần transcript tham chiếu. "
                    "Hãy nhập đúng câu đang có trong audio tham chiếu hoặc đổi về mode `turbo`."
                )
            infer_kwargs["ref_codes"] = voice
            infer_kwargs["ref_text"] = reference_text

        try:
            audio = vieneu.infer(**infer_kwargs)
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"VieNeu sinh audio thất bại: {exc}") from exc

        audio_np = _to_numpy_audio(audio)
        if audio_np.size == 0:
            raise TTSError("VieNeu trả về audio rỗng. Hãy thử đổi audio tham chiếu hoặc rút ngắn đoạn văn.")

        sample_rate = int(getattr(vieneu, "sample_rate", 24000) or 24000)
        sf.write(output_path, audio_np, sample_rate)

        elapsed = time.perf_counter() - started
        estimated_chunks = max(1, len(_split_text_for_tts(text, max_chars=220)))
        notes = [
            "VieNeu clone giọng trực tiếp từ reference embedding, không dùng flow speech-token kiểu ViRa.",
            f"Audio tham chiếu sau chuẩn hóa dài khoảng {ref_duration:.2f}s.",
        ]
        if ref_prep_stats.get("trimmed"):
            notes.append(
                f"Đã tự cắt khoảng lặng đầu/cuối khỏi audio tham chiếu (~{float(ref_prep_stats['trimmed_seconds']):.2f}s)."
            )
        if float(ref_prep_stats.get("activity_ratio_after_trim", 0.0)) < 0.28:
            notes.append("Audio tham chiếu có mật độ giọng nói hơi thưa; VieNeu vẫn chạy được nhưng chất lượng clone có thể giảm.")
        if reference_text and not requires_reference_text:
            notes.append(f"Mode VieNeu `{model_spec['mode_name']}` không bắt buộc transcript tham chiếu; trường này chỉ để bạn lưu đối chiếu.")
        if requires_reference_text:
            notes.append(f"Đã dùng mode VieNeu `{model_spec['mode_name']}` nên transcript tham chiếu được truyền vào model.")
        if model_spec["key"] != "default":
            notes.insert(0, f"Đã dùng mode VieNeu tùy chọn: {model_spec['label']}.")

        return SynthesisResult(
            engine_id="vieneu",
            engine_label="VieNeu-TTS",
            model_key=model_spec["key"],
            model_label=model_spec["label"],
            output_path=output_path,
            sample_rate=sample_rate,
            duration_seconds=len(audio_np) / float(sample_rate),
            inference_seconds=elapsed,
            chunk_count=estimated_chunks,
            reference_text_used=bool(reference_text and requires_reference_text),
            seed=None,
            notes=notes,
        )

    @staticmethod
    def _count_speech_tokens(llm_response: str) -> int:
        """Count valid speech_token_XXX patterns in the LLM output."""
        return len(re.findall(r"speech_token_\d+", llm_response or ""))

    @staticmethod
    def _llm_output_preview(llm_response: str, limit: int = 160) -> str:
        compact = re.sub(r"\s+", " ", llm_response or "").strip()
        if not compact:
            return "trống"
        if len(compact) > limit:
            return compact[:limit].rstrip() + "..."
        return compact

    @staticmethod
    def _normalize_vira_retry_text(
        chunk: str,
        *,
        aggressive: bool = False,
        ensure_terminal: bool = True,
    ) -> str:
        cleaned = _fallback_cleanup_vira_text(chunk, aggressive=False, ensure_terminal=False)
        if not cleaned:
            return ""

        try:
            from mira.utils import normalize_vietnamese, punc_norm
        except Exception:
            normalized = cleaned
        else:
            try:
                normalized = normalize_vietnamese(cleaned)
                normalized = punc_norm(normalized)
            except Exception:
                normalized = cleaned

        return _fallback_cleanup_vira_text(
            normalized,
            aggressive=aggressive,
            ensure_terminal=ensure_terminal,
        )

    def _mutate_chunk_text(self, chunk: str, attempt: int) -> str:
        """Slightly modify chunk text on retries to nudge the LLM.

        Sometimes the LLM gets stuck on certain text patterns and produces
        zero or too few speech tokens. Small cleanup steps that stay close to
        ViRa's own preprocessing help more than adding meta-instructions.
        """
        chunk = re.sub(r"\s+", " ", chunk or "").strip()
        if attempt <= 0:
            return chunk
        if attempt == 1:
            return self._normalize_vira_retry_text(chunk, aggressive=False, ensure_terminal=True) or chunk
        if attempt == 2:
            return self._normalize_vira_retry_text(chunk, aggressive=True, ensure_terminal=True) or chunk
        if attempt == 3:
            return self._normalize_vira_retry_text(chunk, aggressive=True, ensure_terminal=False) or chunk
        normalized = self._normalize_vira_retry_text(chunk, aggressive=True, ensure_terminal=True) or chunk
        words = normalized.split()
        mid = len(words) // 2
        if mid > 0 and "," not in normalized:
            words.insert(mid, ",")
            return " ".join(words)
        return normalized

    def _vira_generate_with_recovery(
        self,
        model: Any,
        chunk: str,
        context_tokens: str,
        *,
        split_depth: int = 0,
        max_split_depth: int = 2,
    ) -> tuple[list[np.ndarray], list[str]]:
        retry_budget = max(3, 5 - split_depth)

        try:
            audio, notes = self._vira_generate_safe(
                model,
                chunk,
                context_tokens,
                max_retries=retry_budget,
                min_speech_tokens=10,
            )
        except TTSError as exc:
            root_error = exc
        else:
            audio_np = _to_numpy_audio(audio)
            if audio_np.size == 0:
                root_error = TTSError("ViRa trả về audio rỗng.")
            else:
                return [audio_np], notes

        if split_depth >= max_split_depth:
            raise root_error

        sub_chunks = _split_failed_vira_chunk(chunk)
        if len(sub_chunks) <= 1:
            raise root_error

        notes = [
            f"Tách fallback depth {split_depth + 1}: chunk được chia thành {len(sub_chunks)} đoạn ngắn hơn."
        ]
        recovered_waves: list[np.ndarray] = []
        recovered_any = False

        for sub_index, sub_chunk in enumerate(sub_chunks):
            try:
                sub_waves, sub_notes = self._vira_generate_with_recovery(
                    model,
                    sub_chunk,
                    context_tokens,
                    split_depth=split_depth + 1,
                    max_split_depth=max_split_depth,
                )
            except TTSError:
                notes.append(
                    f"Sub-chunk {sub_index + 1}/{len(sub_chunks)} thất bại, bị bỏ qua."
                )
                continue

            recovered_any = True
            recovered_waves.extend(sub_waves)
            notes.extend(
                f"Sub-chunk {sub_index + 1}/{len(sub_chunks)}: {note}"
                for note in sub_notes
            )

        if recovered_any:
            return recovered_waves, notes
        raise root_error

    def _vira_generate_safe(
        self,
        model: Any,
        chunk: str,
        context_tokens: str,
        *,
        max_retries: int = 5,
        min_speech_tokens: int = 10,
    ) -> tuple[Any, list[str]]:
        """Generate audio for a single chunk with validation and retry logic.

        The upstream AudioDecoder.detokenize() extracts speech tokens via regex.
        If the LLM produces fewer valid speech_token_XXX patterns than the ONNX
        Conv kernel's minimum receptive field, the resulting tensor crashes the
        ONNX Conv node at '/out_project/Conv' with "Invalid input shape: {0}".

        Defence layers:
          1. Token count validation (min_speech_tokens=10)
          2. Separate try/except around codec.decode() so ONNX errors trigger retry
          3. Temperature + text mutation schedule across retries
          4. Escalating max_new_tokens budget
        """
        codec = model.codec
        pipe = model.pipe
        base_gen_config = model.gen_config
        notes: list[str] = []

        # Retry schedule: temperature, top_k, repetition_penalty, max_new_tokens
        # Gradually increase randomness and output budget.
        retry_schedule = [
            {},                                                                          # 0: original
            {"temperature": 0.9,  "top_k": 80},                                         # 1: warmer
            {"temperature": 1.0,  "top_k": 100, "repetition_penalty": 1.05},             # 2: more diverse
            {"temperature": 0.85, "top_k": 60,  "max_new_tokens": 2048},                 # 3: longer budget, lower temp
            {"temperature": 1.1,  "top_k": 120, "repetition_penalty": 1.0, "max_new_tokens": 2048},  # 4: max explore
        ]

        last_error: Exception | None = None
        last_token_count = 0
        best_token_count = 0
        last_llm_preview = "trống"

        for attempt in range(max_retries):
            # Mutate text on retries to break degenerate LLM patterns
            attempt_chunk = self._mutate_chunk_text(chunk, attempt)

            # Build generation config for this attempt
            overrides = retry_schedule[attempt] if attempt < len(retry_schedule) else retry_schedule[-1]
            if overrides:
                try:
                    from lmdeploy import GenerationConfig
                    gen_config = GenerationConfig(
                        top_p=overrides.get("top_p", base_gen_config.top_p),
                        top_k=overrides.get("top_k", base_gen_config.top_k),
                        temperature=overrides.get("temperature", base_gen_config.temperature),
                        max_new_tokens=overrides.get("max_new_tokens", base_gen_config.max_new_tokens),
                        repetition_penalty=overrides.get("repetition_penalty", base_gen_config.repetition_penalty),
                        do_sample=True,
                        min_p=overrides.get("min_p", base_gen_config.min_p),
                    )
                except Exception:
                    gen_config = base_gen_config
            else:
                gen_config = base_gen_config

            # --- Step 1: LLM inference ---
            try:
                formatted_prompt = codec.format_prompt(attempt_chunk, context_tokens, None)
                response = pipe([formatted_prompt], gen_config=gen_config, do_preprocess=False)
                llm_output = response[0].text if response else ""
            except Exception as exc:
                last_error = exc
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries} LLM lỗi: {type(exc).__name__}: {exc}"
                )
                continue

            # --- Step 2: Validate token count ---
            last_llm_preview = self._llm_output_preview(llm_output)
            token_count = self._count_speech_tokens(llm_output)
            last_token_count = token_count
            best_token_count = max(best_token_count, token_count)

            if token_count < min_speech_tokens:
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries}: LLM trả về {token_count} speech token "
                    f"(cần ≥{min_speech_tokens})."
                )
                continue  # retry with next config

            # --- Step 3: Decode tokens to audio (separate try/except for ONNX) ---
            try:
                audio = codec.decode(llm_output, context_tokens)
            except Exception as exc:
                last_error = exc
                notes.append(
                    f"Chunk retry {attempt + 1}/{max_retries} decode lỗi ({token_count} tokens): "
                    f"{type(exc).__name__}: {exc}"
                )
                # Raise min threshold for next attempt since tokens were insufficient
                min_speech_tokens = max(min_speech_tokens, token_count + 5)
                continue

            # --- Step 4: Apply fade in/out ---
            try:
                from mira.model import apply_fade_in_out
                sample_rate = 48000
                fade_in_samples = int(10 * sample_rate / 1000)
                fade_out_samples = int(50 * sample_rate / 1000)
                audio = apply_fade_in_out(audio, fade_in_samples, fade_out_samples)
            except Exception:
                pass  # fade is cosmetic, don't fail on it

            if attempt > 0:
                notes.append(f"Chunk thành công sau {attempt + 1} lần thử.")
            return audio, notes

        # All retries exhausted
        last_error_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "không có exception từ model, nhưng LLM không sinh speech token hợp lệ"
        )
        llm_preview_text = (
            "LLM output cuối trống."
            if last_llm_preview == "trống"
            else f"LLM output cuối: {last_llm_preview}."
        )
        raise TTSError(
            f"ViRa sinh audio thất bại sau {max_retries} lần thử. "
            f"LLM trả tối đa {best_token_count} speech token; lần cuối là {last_token_count} (cần ≥{min_speech_tokens}). "
            f"Lỗi cuối: {last_error_text}. {llm_preview_text} "
            "Nguyên nhân thường là audio tham chiếu quá ngắn, có khoảng lặng lớn, "
            "hoặc chunk văn bản khiến model không sinh được token hợp lệ. "
            "Hãy thử: (1) đổi audio tham chiếu dài 3-10s, nói rõ; "
            "(2) rút ngắn câu văn; (3) tách thành nhiều câu ngắn hơn."
        )

    def _synthesize_with_vira(
        self,
        *,
        text: str,
        reference_audio: Path,
        reference_text: str,
        model_spec: dict[str, Any],
        output_path: Path,
    ) -> SynthesisResult:
        model = self._load_vira(model_spec)
        try:
            from mira.utils import split_text as vira_split_text
        except Exception:
            vira_split_text = None

        context_started = time.perf_counter()
        prepared_reference_audio, ref_duration, ref_prep_stats = self._prepare_reference_audio_for_vira(reference_audio)
        try:
            context_tokens = model.encode_audio(str(prepared_reference_audio))
        except Exception as exc:  # pragma: no cover - depends on external install
            raise TTSError(f"ViRa không encode được audio tham chiếu: {exc}") from exc

        context_numel = _numel(context_tokens)
        if context_numel is not None and context_numel <= 0:
            raise TTSError(
                "ViRa encode ra context token rỗng từ audio tham chiếu. Hãy đổi sang WAV mono, 48kHz hoặc dùng đoạn nói 3-10 giây rõ hơn."
            )

        ctx_token_count: int | None = context_numel
        # Also validate context_tokens is a non-empty string with actual token patterns
        if isinstance(context_tokens, str):
            ctx_token_count = len(re.findall(r"context_token_\d+", context_tokens))
            if ctx_token_count == 0:
                raise TTSError(
                    "ViRa encode ra context token string rỗng (0 context_token pattern). "
                    "Audio tham chiếu có thể quá ngắn hoặc gần im lặng. "
                    "Hãy dùng WAV mono, 3-10 giây nói rõ ràng."
                )

        base_chunks = vira_split_text(text) if vira_split_text else [text]
        if isinstance(base_chunks, str):
            base_chunks = [base_chunks]
        chunks: list[str] = []
        for base_chunk in base_chunks:
            chunks.extend(_split_text_for_tts(base_chunk, max_chars=140))
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            raise TTSError("Không có câu hợp lệ để ViRa sinh audio.")

        chunk_waves: list[np.ndarray] = []
        all_notes: list[str] = []
        skipped_chunks: list[int] = []

        for index, chunk in enumerate(chunks):
            try:
                recovered_waves, chunk_notes = self._vira_generate_with_recovery(
                    model, chunk, context_tokens,
                )
                all_notes.extend(
                    f"Chunk {index + 1}/{len(chunks)}: {note}"
                    for note in chunk_notes
                )
            except TTSError as exc:
                if len(chunks) == 1:
                    activity = float(ref_prep_stats.get("activity_ratio_after_trim", ref_prep_stats.get("activity_ratio", 0.0)))
                    ctx_hint = str(ctx_token_count) if ctx_token_count is not None else "không rõ"
                    raise TTSError(
                        f"{exc} Audio tham chiếu sau chuẩn hóa dài {ref_duration:.2f}s, "
                        f"mật độ giọng khoảng {activity:.2f}, context token {ctx_hint}."
                    ) from exc
                skipped_chunks.append(index + 1)
                all_notes.append(
                    f"Chunk {index + 1}/{len(chunks)} bị bỏ qua do LLM không sinh được speech token hợp lệ."
                )
                continue

            if not recovered_waves:
                if len(chunks) == 1:
                    raise TTSError(
                        f"ViRa trả về audio rỗng ở chunk {index + 1}. "
                        "Hãy thử rút ngắn câu, đổi audio tham chiếu sang WAV mono rõ tiếng, "
                        "hoặc tách đoạn văn thành các câu ngắn hơn."
                    )
                skipped_chunks.append(index + 1)
                all_notes.append(f"Chunk {index + 1}/{len(chunks)} trả về audio rỗng, bị bỏ qua.")
                continue

            chunk_waves.extend(recovered_waves)

        if not chunk_waves:
            raise TTSError(
                "ViRa không sinh được audio cho bất kỳ chunk nào. "
                "Hãy thử: (1) đổi audio tham chiếu dài 3-10s, nói rõ; "
                "(2) rút ngắn câu văn; (3) dùng audio WAV mono 48kHz."
            )

        audio_np = _crossfade_join(chunk_waves, sample_rate=48000, duration_ms=65)
        sample_rate = 48000
        sf.write(output_path, audio_np, sample_rate)

        elapsed = time.perf_counter() - context_started
        notes = [
            "ViRa dùng reference audio đã được chuẩn hóa về WAV mono 48kHz và biên độ ổn định trước khi encode context token.",
            f"Audio tham chiếu sau chuẩn hóa dài khoảng {ref_duration:.2f}s.",
        ]
        if ref_prep_stats.get("trimmed"):
            notes.append(
                f"Đã tự cắt khoảng lặng đầu/cuối khỏi audio tham chiếu (~{float(ref_prep_stats['trimmed_seconds']):.2f}s)."
            )
        if float(ref_prep_stats.get("activity_ratio_after_trim", 0.0)) < 0.35:
            notes.append("Audio tham chiếu có mật độ giọng nói khá thưa; ViRa có thể kém ổn định hơn bình thường.")
        if len(chunks) > 1:
            notes.append(f"ViRa đã sinh tuần tự {len(chunks)} chunk rồi crossfade để giảm lỗi codec và rớt shape.")
        if skipped_chunks:
            notes.append(f"⚠️ Đã bỏ qua {len(skipped_chunks)} chunk lỗi (chunk {', '.join(map(str, skipped_chunks))}).")
        if reference_text:
            notes.append("ViRa hiện không dùng transcript tham chiếu; trường này chỉ để người dùng đối chiếu.")
        if model_spec["key"] != "default":
            notes.insert(0, f"Đã dùng model ViRa tùy chọn: {model_spec['label']}.")
        notes.extend(all_notes)

        return SynthesisResult(
            engine_id="vira",
            engine_label="ViRa",
            model_key=model_spec["key"],
            model_label=model_spec["label"],
            output_path=output_path,
            sample_rate=sample_rate,
            duration_seconds=len(audio_np) / float(sample_rate),
            inference_seconds=elapsed,
            chunk_count=len(chunks),
            reference_text_used=False,
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
            return "Gwen-TTS trong project này cần GPU CUDA để chạy local. Nếu runtime chỉ có CPU, hãy dùng VieNeu hoặc F5."

        return None

    def _vieneu_runtime_stack_warning(self, mode_name: str) -> str | None:
        if not _vieneu_mode_requires_reference_text(mode_name):
            return None

        version_label = self._torch_version_label()
        if not version_label:
            return (
                f"VieNeu {mode_name} cần stack PyTorch GPU của upstream. "
                "Notebook hiện tại phải cài lại `vieneu[gpu]` và torch/torchaudio mới hơn trước khi generate."
            )

        match = re.match(r"^\s*(\d+)\.(\d+)", version_label)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            if (major, minor) < (2, 11):
                return (
                    f"VieNeu {mode_name} cần torch >= 2.11 theo stack hiện tại của upstream, "
                    f"nhưng runtime đang có torch {version_label}. "
                    "Rerun notebook mới để cài lại torch/torchaudio theo CUDA 12.8."
                )

        return None
