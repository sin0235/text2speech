"""
webapp/scraper.py — Google Maps Review Scraper via Apify

Hỗ trợ 2 chế độ:
  1. URL mode  : nhập trực tiếp link Google Maps
  2. Query mode: nhập search query + location
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Apify actor ID cho Google Maps Scraper
ACTOR_ID = "nwua9Gu5YrADL7ZDj"


def _load_dotenv_project_root():
    """Đảm bảo .env (UTF-8 BOM ok) được nạp khi scraper import động."""
    try:
        from dotenv import load_dotenv

        load_dotenv(
            Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig",
        )
    except ImportError:
        pass


def _get_client():
    """Khởi tạo ApifyClient với token từ biến môi trường APIFY_API_TOKEN."""
    _load_dotenv_project_root()
    try:
        from apify_client import ApifyClient
    except ImportError:
        raise ImportError(
            "apify-client chưa được cài. Hãy chạy: pip install apify-client"
        )
    token = (os.environ.get("APIFY_API_TOKEN") or "").strip()
    if (len(token) >= 2) and ((token[0] == token[-1]) and token[0] in "\"'"):
        token = token[1:-1].strip()
    if not token:
        raise ValueError(
            "Thiếu Apify API token. Tạo token tại console.apify.com, rồi đặt "
            "APIFY_API_TOKEN (hoặc thêm vào file .env ở thư mục gốc project)."
        )
    return ApifyClient(token)


def _parse_review(item: dict) -> dict:
    """Chuẩn hóa một review từ Apify output thành dict dùng cho webapp."""
    # Lấy text review
    text = (
        item.get("text")
        or item.get("reviewText")
        or item.get("snippet")
        or ""
    ).strip()

    # Rating (1-5 sao)
    rating = item.get("stars") or item.get("rating") or item.get("reviewRating")
    try:
        rating = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None

    # Ngày đăng
    publish_at = (
        item.get("publishAt")
        or item.get("reviewDate")
        or item.get("time")
        or ""
    )
    # Normalize ngày
    date_str = ""
    if publish_at:
        try:
            # Apify trả về ISO format hoặc relative string
            if "T" in str(publish_at):
                dt = datetime.fromisoformat(str(publish_at).replace("Z", "+00:00"))
                date_str = dt.strftime("%d/%m/%Y")
            else:
                date_str = str(publish_at)
        except Exception:
            date_str = str(publish_at)

    # Tác giả
    reviewer = (
        item.get("reviewerName")
        or item.get("name")
        or item.get("authorName")
        or "Ẩn danh"
    )

    # Số lượt thích
    likes = item.get("likesCount") or item.get("reviewLikes") or 0

    # URL ảnh đại diện
    avatar = item.get("reviewerPhotoUrl") or item.get("reviewerUrl") or ""

    return {
        "text": text,
        "rating": rating,
        "date": date_str,
        "reviewer": reviewer,
        "likes": likes,
        "avatar": avatar,
        "raw": item,  # giữ original để debug
    }


def scrape_google_maps(
    *,
    url: Optional[str] = None,
    query: Optional[str] = None,
    location: Optional[str] = None,
    max_reviews: int = 25,
    max_places: int = 1,
    language: str = "en",
    reviews_sort: str = "newest",
    reviews_filter: str = "",
    reviews_origin: str = "all",
    reviews_start_date: Optional[str] = None,
    search_matching: str = "all",
    place_min_stars: str = "",
    skip_closed: bool = False,
    scrape_reviews_personal_data: bool = True,
) -> dict:
    """
    Cào review từ Google Maps qua Apify.

    Args:
        url           : Link Google Maps cụ thể
        query         : Tên nhà hàng/địa điểm (nếu không có url)
        location      : Địa điểm tìm kiếm (VD: "Ho Chi Minh City, Vietnam")
        max_reviews   : Số review tối đa cần lấy
        max_places    : Số địa điểm tối đa khi tìm kiếm
        language      : Ngôn ngữ (en / vi / ...)
        reviews_sort  : Cách sắp xếp (newest / highestRanking / lowestRanking / relevant)
        reviews_filter: Lọc review theo từ khoá
        reviews_origin: Nguồn review (all / googleReviews)
        reviews_start_date: Chỉ lấy review từ ngày này (YYYY-MM-DD)
        search_matching: Matching mode (all / exact)
        place_min_stars: Số sao tối thiểu ("" / "1" - "5")
        skip_closed   : Bỏ qua địa điểm đóng cửa
        scrape_reviews_personal_data: Lấy thông tin cá nhân reviewer

    Returns:
        {
          "ok": bool,
          "reviews": list[dict],
          "place_name": str,
          "place_address": str,
          "total_score": float | None,
          "total_reviews": int,
          "run_id": str,
          "error": str  # chỉ khi ok=False
        }
    """
    if not url and not query:
        return {"ok": False, "error": "Cần cung cấp URL hoặc search query."}

    try:
        client = _get_client()
    except (ImportError, ValueError) as e:
        return {"ok": False, "error": str(e)}

    # Build start_urls nếu có URL trực tiếp
    start_urls = None
    search_strings = None

    if url:
        # Validate URL: phải là Google Maps
        if not re.search(r"(google\.(com|[a-z]{2})\/maps|maps\.google|goo\.gl)", url):
            return {
                "ok": False,
                "error": "URL không hợp lệ. Vui lòng nhập link Google Maps.",
            }
        start_urls = [{"url": url}]
    else:
        search_strings = [query]

    run_input = {
        "locationQuery": location or "",
        "maxCrawledPlacesPerSearch": max_places,
        "maxReviews": max_reviews,
        "reviewsSort": reviews_sort,
        "reviewsFilterString": reviews_filter or "",
        "reviewsOrigin": reviews_origin or "all",
        "language": language,
        "searchMatching": search_matching or "all",
        "placeMinimumStars": place_min_stars or "",
        "skipClosedPlaces": skip_closed,
        "scrapePlaceDetailPage": False,
        "scrapeContacts": False,
        "scrapeTableReservationProvider": False,
        "includeWebResults": False,
        "scrapeDirectories": False,
        "maxImages": 0,
        "maxQuestions": 0,
        "scrapeReviewsPersonalData": scrape_reviews_personal_data,
        "scrapeImageAuthors": False,
        "scrapeSocialMediaProfiles": {
            "facebooks": False,
            "instagrams": False,
            "youtubes": False,
            "tiktoks": False,
            "twitters": False,
        },
        "maximumLeadsEnrichmentRecords": 0,
        "website": "allPlaces",
    }
    if reviews_start_date:
        run_input["reviewsStartDate"] = reviews_start_date
    if start_urls:
        run_input["startUrls"] = start_urls
    if search_strings:
        run_input["searchStringsArray"] = search_strings

    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "not valid" in msg or "authentication" in msg:
            return {
                "ok": False,
                "error": (
                    "Apify từ chối token (sai, hết hạn hoặc đã thu hồi). "
                    "Vào console.apify.com → Settings → Integrations → API tokens, "
                    "tạo token mới (dạng apify_api_...), ghi vào .env: APIFY_API_TOKEN=..., "
                    "đặt file .env ở thư mục gốc project (cùng cấp webapp), khởi động lại Flask. "
                    f"Chi tiết: {e}"
                ),
            }
        return {"ok": False, "error": f"Lỗi khi gọi Apify API: {e}"}

    if not run:
        return {"ok": False, "error": "Apify run thất bại (run=None)."}

    # Thu thập kết quả
    reviews = []
    place_name = ""
    place_address = ""
    total_score = None
    place_total_reviews = 0

    try:
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            # Mỗi item có thể là info của địa điểm (chứa reviews lồng trong đó)
            name = item.get("title") or item.get("name") or ""
            address = (
                item.get("address")
                or item.get("street")
                or item.get("neighborhood")
                or ""
            )
            score = item.get("totalScore") or item.get("rating")
            n_reviews_place = item.get("reviewsCount") or item.get("userRatingsTotal") or 0

            if name and not place_name:
                place_name = name
            if address and not place_address:
                place_address = address
            if score and total_score is None:
                try:
                    total_score = round(float(score), 1)
                except (TypeError, ValueError):
                    pass
            if n_reviews_place and not place_total_reviews:
                try:
                    place_total_reviews = int(n_reviews_place)
                except (TypeError, ValueError):
                    pass

            # Reviews có thể là field lồng nhau
            nested_reviews = item.get("reviews") or []
            if nested_reviews:
                for rv in nested_reviews:
                    parsed = _parse_review(rv)
                    if parsed["text"]:
                        reviews.append(parsed)
            else:
                # Hoặc item chính là review
                if item.get("text") or item.get("reviewText"):
                    parsed = _parse_review(item)
                    if parsed["text"]:
                        reviews.append(parsed)
    except Exception as e:
        return {"ok": False, "error": f"Lỗi khi đọc dataset: {e}"}

    return {
        "ok": True,
        "reviews": reviews,
        "place_name": place_name,
        "place_address": place_address,
        "total_score": total_score,
        "total_reviews": place_total_reviews,
        "run_id": run.get("id", ""),
    }
