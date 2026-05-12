import threading
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests


AGRO_NEWS_URL = "https://g1.globo.com/rss/g1/economia/agronegocios/"
CACHE_TTL_SECONDS = 15 * 60
REQUEST_TIMEOUT_SECONDS = 10
MAX_ITEMS = 5

_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {
    "updated_at": 0.0,
    "data": None,
}


def _clean_text(value: Optional[str]) -> str:
    return " ".join((value or "").split())


def _format_pub_date(value: str) -> Optional[str]:
    if not value:
        return None

    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return value


def _parse_rss(xml_text: str) -> List[Dict[str, Optional[str]]]:
    root = ET.fromstring(xml_text)
    items = []

    for item in root.findall("./channel/item")[:MAX_ITEMS]:
        items.append(
            {
                "title": _clean_text(item.findtext("title")),
                "link": _clean_text(item.findtext("link")),
                "published_at": _format_pub_date(_clean_text(item.findtext("pubDate"))),
                "summary": _clean_text(item.findtext("description")),
            }
        )

    return items


def scrape_agro_news(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Busca noticias publicas do agronegocio e guarda cache simples.
    O cache evita varias requisicoes seguidas na mesma fonte.
    """
    now = time.time()

    with _cache_lock:
        cached_data = _cache["data"]
        cache_age = now - float(_cache["updated_at"])

        if cached_data and not force_refresh and cache_age < CACHE_TTL_SECONDS:
            return {
                **cached_data,
                "from_cache": True,
                "cache_expires_in_seconds": int(CACHE_TTL_SECONDS - cache_age),
            }

    try:
        response = requests.get(
            AGRO_NEWS_URL,
            headers={"User-Agent": "AgroVision/1.0 academic project"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        news_items = _parse_rss(response.text)

        data = {
            "source": "G1 Agronegocios RSS",
            "source_url": AGRO_NEWS_URL,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": news_items,
            "error": None,
        }

        with _cache_lock:
            _cache["updated_at"] = now
            _cache["data"] = data

        return {
            **data,
            "from_cache": False,
            "cache_expires_in_seconds": CACHE_TTL_SECONDS,
        }
    except (requests.RequestException, ET.ParseError) as exc:
        with _cache_lock:
            cached_data = _cache["data"]

        if cached_data:
            return {
                **cached_data,
                "from_cache": True,
                "cache_expires_in_seconds": 0,
                "error": f"Fonte indisponivel, exibindo ultimo cache: {exc}",
            }

        return {
            "source": "G1 Agronegocios RSS",
            "source_url": AGRO_NEWS_URL,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": [],
            "from_cache": False,
            "cache_expires_in_seconds": 0,
            "error": f"Nao foi possivel coletar noticias agro: {exc}",
        }
