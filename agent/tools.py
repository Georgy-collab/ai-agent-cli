"""Инструменты агента: поиск, HTTP, файлы, память, погода, криптовалюта, оболочка."""

from __future__ import annotations

import html
import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
from langchain_core.tools import tool

logger = logging.getLogger("agent.tools")

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
# Единственная папка, откуда агент читает рабочие файлы и куда их записывает.
FILES_DIR = WORKSPACE / "files"
MEMORY_PATH = AGENT_DIR / "memory.json"

HTTP_TIMEOUT = 20
HTTP_MAX_CHARS = 12_000
SHELL_TIMEOUT = 30
SHELL_MAX_CHARS = 8_000
SEARCH_MAX_RESULTS = 5

USER_AGENT = "local-cli-agent/1.0 (educational local agent)"
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Файлы, которые агент не должен читать или перезаписывать (там ключи).
_SENSITIVE_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "credentials.json"}

_COIN_ALIASES = {
    "btc": "bitcoin",
    "биткоин": "bitcoin",
    "биток": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "эфир": "ethereum",
    "эфириум": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "солана": "solana",
    "solana": "solana",
    "ton": "the-open-network",
    "тон": "the-open-network",
    "doge": "dogecoin",
    "доги": "dogecoin",
    "dogecoin": "dogecoin",
    "xrp": "ripple",
    "рипл": "ripple",
    "ada": "cardano",
    "кардано": "cardano",
    "ltc": "litecoin",
    "лайткоин": "litecoin",
    "bnb": "binancecoin",
    "usdt": "tether",
    "тезер": "tether",
}

_CURRENCY_ALIASES = {
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "доллар": "usd",
    "доллары": "usd",
    "$": "usd",
    "eur": "eur",
    "euro": "eur",
    "евро": "eur",
    "rub": "rub",
    "rur": "rub",
    "руб": "rub",
    "рубль": "rub",
    "рубли": "rub",
    "рублей": "rub",
    "kzt": "kzt",
    "тенге": "kzt",
    "gbp": "gbp",
    "фунт": "gbp",
    "uah": "uah",
    "гривна": "uah",
    "cny": "cny",
    "юань": "cny",
}

# Коды погоды WMO, которые отдаёт Open-Meteo в current_weather.weathercode.
_WMO_WEATHER = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "туман с изморозью",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "гроза с сильным градом",
}

# Базовая защита локальной оболочки: отказ на заведомо разрушительных командах.
_DANGEROUS_SHELL = re.compile(
    r"("
    r"\bformat(?:-volume)?\b"
    r"|\bshutdown\b"
    r"|\bmkfs\b"
    r"|\bdiskpart\b"
    r"|\bcipher\s+/w"
    r"|\bdel\s+/[sq]"
    r"|\brd\s+/s"
    r"|\brm\s+-[a-z]*r"
    r"|remove-item\b[^\n]*-recurse"
    r"|clear-disk"
    r"|stop-computer"
    r"|restart-computer"
    r"|\breg\s+delete\b"
    r"|\bbcdedit\b"
    r"|:\(\)\s*\{"
    r")",
    re.IGNORECASE,
)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [обрезано, всего {len(text)} символов]"


def _is_sensitive(path: Path) -> bool:
    return path.name.lower() in _SENSITIVE_NAMES or path.name.lower().startswith(".env")


def _files_relative(raw: str) -> str:
    """Убирает префикс files/, если модель передала путь уже с именем папки."""
    text = raw.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    lowered = text.lower()
    if lowered == "files":
        return "."
    if lowered.startswith("files/"):
        return text[6:] or "."
    return text


def _safe_path(raw: str) -> Path:
    """Разрешает путь только внутри папки files."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    if not raw or not str(raw).strip():
        raise ValueError("Путь пустой.")
    relative = _files_relative(str(raw))
    candidate = Path(relative)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (FILES_DIR / candidate).resolve()
    root = FILES_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Читать и записывать файлы можно только в папке {root}")
    if _is_sensitive(resolved):
        raise ValueError("Доступ к файлам с секретами запрещён.")
    return resolved


def _empty_memory() -> dict[str, Any]:
    return {"notes": [], "conversations": []}


def load_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return _empty_memory()
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("memory.json повреждён (%s), начинаю с пустой памяти", exc)
        return _empty_memory()
    if not isinstance(data, dict):
        return _empty_memory()
    data.setdefault("notes", [])
    data.setdefault("conversations", [])
    return data


def save_memory(data: dict[str, Any]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temporary = MEMORY_PATH.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(MEMORY_PATH)


def ensure_storage() -> None:
    """Создаёт папку files и пустой memory.json, если их ещё нет. Существующие данные не трогает."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        save_memory(_empty_memory())


def memory_context(limit: int = 8) -> str:
    """Короткий текст последних диалогов и заметок для системного промпта."""
    data = load_memory()
    lines: list[str] = []
    notes = data.get("notes") or []
    if notes:
        lines.append("Заметки:")
        for note in notes[-10:]:
            lines.append(f"- {note.get('text', '')}")
    conversations = data.get("conversations") or []
    if conversations:
        lines.append("Недавние диалоги:")
        for item in conversations[-limit:]:
            lines.append(f"- {item.get('summary', '')}")
    return "\n".join(lines).strip()


def append_conversation(user_text: str, assistant_text: str, tools_used: list[str]) -> None:
    data = load_memory()
    tools = ", ".join(tools_used) if tools_used else "без инструментов"
    summary = (
        f"Пользователь: {_clip(user_text.strip(), 300)} | "
        f"Ответ: {_clip(assistant_text.strip(), 500)} | "
        f"Инструменты: {tools}"
    )
    data["conversations"].append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "user": user_text.strip()[:2000],
            "assistant": assistant_text.strip()[:4000],
            "tools": tools_used,
            "summary": summary,
        }
    )
    data["conversations"] = data["conversations"][-50:]
    save_memory(data)
    logger.info("Диалог сохранён в memory.json (инструменты: %s)", tools)


def _normalize_coin(coin: str) -> str:
    key = (coin or "").strip().lower().replace(" ", "")
    if key in _COIN_ALIASES:
        return _COIN_ALIASES[key]
    return (coin or "").strip().lower().replace(" ", "-")


def _normalize_currency(currency: str) -> str:
    key = (currency or "").strip().lower()
    return _CURRENCY_ALIASES.get(key, key or "usd")


def get_crypto_price(coin: str, currency: str = "usd") -> float:
    """Курс монеты в выбранной валюте через CoinGecko. Возвращает число."""
    coin_id = _normalize_coin(coin)
    vs = _normalize_currency(currency)
    if not coin_id:
        raise ValueError("Не указана криптовалюта.")
    if not re.fullmatch(r"[a-z0-9\-]+", coin_id):
        raise ValueError(f"Непонятное имя монеты: {coin}")
    if not re.fullmatch(r"[a-z]{3,5}", vs):
        raise ValueError(f"Непонятная валюта: {currency}. Пример: usd, eur, rub.")

    response = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": coin_id, "vs_currencies": vs},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code == 429:
        raise RuntimeError("CoinGecko временно ограничил запросы. Повторите через минуту.")
    response.raise_for_status()
    payload = response.json()
    if coin_id not in payload or vs not in payload[coin_id]:
        raise ValueError(
            f"Курс не найден для «{coin}» ({coin_id}) в валюте {vs}. "
            "Уточните id монеты, например bitcoin или ethereum."
        )
    return float(payload[coin_id][vs])


def _geocode(city: str) -> dict[str, Any]:
    last_error = "Город не найден."
    for language in ("ru", "en"):
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": language, "format": "json"},
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if results:
            return results[0]
        last_error = f"Город «{city}» не найден."
    raise ValueError(last_error + " Уточните название.")


def _weather_text(code: int | None) -> str:
    if code is None:
        return "нет данных"
    return _WMO_WEATHER.get(int(code), f"код {code}")


def _plain(value: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _search_duckduckgo(query: str) -> list[dict[str, str]]:
    """HTML-выдача DuckDuckGo. Код 202 — антибот-страница без результатов, её пропускаем."""
    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=SEARCH_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.info("web_search duckduckgo: %s", exc)
        return []
    if response.status_code != 200 or "result__a" not in response.text:
        logger.info("web_search duckduckgo: HTTP %s, без выдачи", response.status_code)
        return []
    pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>',
        re.S,
    )
    rows: list[dict[str, str]] = []
    for href, title, snippet in pattern.findall(response.text):
        rows.append(
            {
                "title": _plain(title),
                "url": html.unescape(href),
                "snippet": _plain(snippet),
                "source": "DuckDuckGo",
            }
        )
        if len(rows) >= SEARCH_MAX_RESULTS:
            break
    return rows


def _search_google_news(query: str) -> list[dict[str, str]]:
    """Свежие заголовки. Ссылка ведёт на карточку Google Новостей, текст — в title и snippet."""
    try:
        response = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "ru", "gl": "RU", "ceid": "RU:ru"},
            headers={**SEARCH_HEADERS, "User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as exc:
        logger.info("web_search news: %s", exc)
        return []
    rows: list[dict[str, str]] = []
    for item in root.findall(".//item")[:SEARCH_MAX_RESULTS]:
        source = item.find("source")
        source_name = source.text.strip() if source is not None and source.text else "Google Новости"
        rows.append(
            {
                "title": _plain(item.findtext("title") or ""),
                "url": (item.findtext("link") or "").strip(),
                "snippet": _plain(item.findtext("description") or ""),
                "published": (item.findtext("pubDate") or "").strip(),
                "source": source_name,
            }
        )
    return [row for row in rows if row["title"]]


def _search_wikipedia(query: str) -> list[dict[str, str]]:
    """Краткое содержание самой близкой статьи. Сначала русская Википедия, затем английская."""
    headers = {**HTTP_HEADERS, "User-Agent": USER_AGENT}
    for host, label in (
        ("https://ru.wikipedia.org", "Википедия"),
        ("https://en.wikipedia.org", "Wikipedia"),
    ):
        try:
            found = requests.get(
                f"{host}/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                    "format": "json",
                    "utf8": 1,
                },
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            found.raise_for_status()
            hits = found.json().get("query", {}).get("search") or []
            if not hits:
                continue
            title = hits[0].get("title") or ""
            summary = requests.get(
                f"{host}/api/rest_v1/page/summary/{quote(title, safe='')}",
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            summary.raise_for_status()
            payload = summary.json()
        except (requests.RequestException, ValueError) as exc:
            logger.info("web_search %s: %s", label, exc)
            continue
        extract = (payload.get("extract") or "").strip()
        if not extract:
            continue
        return [
            {
                "title": payload.get("title") or title,
                "url": (payload.get("content_urls") or {}).get("desktop", {}).get("page")
                or f"{host}/wiki/{quote(title.replace(' ', '_'), safe='')}",
                "snippet": extract,
                "source": label,
            }
        ]
    return []


@tool
def web_search(query: str) -> str:
    """Найти информацию в интернете: новости и справочные статьи.

    Используй для новостей, фактов, документации и всего, чего нет в остальных инструментах.
    query — поисковая фраза на языке пользователя.
    """
    query = (query or "").strip()
    if not query:
        return "Нужна поисковая фраза."
    logger.info("web_search: %s", query)
    rows = _search_duckduckgo(query)
    if rows:
        logger.info("web_search: DuckDuckGo, результатов %s", len(rows))
    else:
        rows = _search_google_news(query) + _search_wikipedia(query)
        logger.info("web_search: новости и энциклопедия, результатов %s", len(rows))
    if not rows:
        return "Поиск ничего не нашёл. Уточните запрос."
    return json.dumps(rows[: SEARCH_MAX_RESULTS + 1], ensure_ascii=False, indent=2)


@tool
def http_request(method: str, url: str, body: str = "", headers_json: str = "") -> str:
    """Выполнить HTTP-запрос к внешнему API через библиотеку requests.

    method: GET, POST, PUT, PATCH или DELETE.
    url: адрес http или https.
    body: необязательная строка (обычно JSON) для POST/PUT/PATCH.
    headers_json: необязательный JSON-объект с заголовками.
    """
    method = (method or "GET").strip().upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        return "Метод должен быть GET, POST, PUT, PATCH или DELETE."
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "URL должен начинаться с http:// или https://."
    headers = dict(HTTP_HEADERS)
    if headers_json.strip():
        try:
            extra = json.loads(headers_json)
        except json.JSONDecodeError:
            return "headers_json должен быть объектом JSON."
        if not isinstance(extra, dict):
            return "headers_json должен быть объектом JSON."
        headers.update({str(key): str(value) for key, value in extra.items()})

    logger.info("http_request %s %s", method, url)
    try:
        response = requests.request(
            method,
            url,
            data=body.encode("utf-8") if body else None,
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("http_request error: %s", exc)
        return f"Запрос не выполнен: {exc}"

    text = _clip(response.text, HTTP_MAX_CHARS)
    return f"HTTP {response.status_code}\n{text}"


@tool
def read_file(path: str) -> str:
    """Прочитать текстовый файл из папки files. path — имя файла или путь внутри files, например notes.txt."""
    try:
        target = _safe_path(path)
    except ValueError as exc:
        return str(exc)
    if not target.is_file():
        return f"Файл не найден: {target}"
    logger.info("read_file %s", target)
    try:
        return _clip(target.read_text(encoding="utf-8", errors="replace"), HTTP_MAX_CHARS)
    except OSError as exc:
        return f"Не удалось прочитать файл: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Записать текст в файл внутри папки files. Другие каталоги недоступны. Создаёт промежуточные папки и перезаписывает файл целиком."""
    try:
        target = _safe_path(path)
    except ValueError as exc:
        return str(exc)
    logger.info("write_file %s (%s символов)", target, len(content or ""))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content or "", encoding="utf-8")
    except OSError as exc:
        return f"Не удалось записать файл: {exc}"
    return f"Записано: {target} ({len(content or '')} символов)"


@tool
def list_directory(path: str = ".") -> str:
    """Показать файлы и папки внутри files. path — подкаталог относительно files, по умолчанию сама папка files."""
    try:
        target = _safe_path(path or ".")
    except ValueError as exc:
        return str(exc)
    if not target.exists():
        return f"Каталог не найден: {target}"
    if not target.is_dir():
        return f"Это не каталог: {target}"
    logger.info("list_directory %s", target)
    entries = []
    for item in sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
        if _is_sensitive(item):
            continue
        kind = "dir" if item.is_dir() else "file"
        entries.append(f"{kind}\t{item.name}")
    if not entries:
        return "Каталог пуст."
    return "\n".join(entries[:200])


@tool
def save_note(text: str) -> str:
    """Сохранить факт в долговременную память (memory.json), чтобы помнить его в следующих сессиях."""
    text = (text or "").strip()
    if not text:
        return "Нечего сохранять."
    data = load_memory()
    data["notes"].append({"at": datetime.now(timezone.utc).isoformat(), "text": text[:1000]})
    data["notes"] = data["notes"][-30:]
    save_memory(data)
    logger.info("save_note")
    return "Заметка сохранена."


@tool
def read_notes() -> str:
    """Прочитать сохранённые заметки и краткие резюме прошлых диалогов из memory.json."""
    data = load_memory()
    return json.dumps(
        {"notes": data.get("notes", [])[-15:], "conversations": data.get("conversations", [])[-8:]},
        ensure_ascii=False,
        indent=2,
    )


@tool
def get_weather(city: str) -> str:
    """Текущая погода в городе: температура, ветер и условия.

    Сначала город превращается в координаты (геокодер Open-Meteo),
    затем запрашивается прогноз. Примеры: «Москва», «Saint Petersburg», «Berlin».
    Если город неочевиден — не угадывай, инструмент вернёт просьбу уточнить.
    """
    city = (city or "").strip()
    if not city:
        return "Нужно название города. Уточните, пожалуйста."
    logger.info("get_weather: %s", city)
    try:
        place = _geocode(city)
        latitude = place["latitude"]
        longitude = place["longitude"]
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
                "timezone": "auto",
                "wind_speed_unit": "kmh",
            },
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        current = response.json().get("current_weather") or {}
    except ValueError as exc:
        return str(exc)
    except requests.RequestException as exc:
        logger.warning("weather error: %s", exc)
        return f"Сервис погоды недоступен: {exc}"

    code = current.get("weathercode")
    place_name = ", ".join(
        part
        for part in (place.get("name"), place.get("admin1"), place.get("country"))
        if part
    )
    return "\n".join(
        [
            f"Город: {place_name}",
            f"Координаты: {latitude}, {longitude}",
            f"Время наблюдения: {current.get('time', 'нет данных')}",
            f"Температура: {current.get('temperature', 'нет данных')} °C",
            f"Ветер: {current.get('windspeed', 'нет данных')} км/ч, направление {current.get('winddirection', 'нет данных')}°",
            f"Условия: {_weather_text(code)}",
            "Источник: Open-Meteo",
        ]
    )


@tool
def crypto_price(coin: str, currency: str = "usd") -> str:
    """Узнать спотовую цену криптовалюты через CoinGecko.

    coin: bitcoin, ethereum, solana или обиходное имя («биткоин», «эфир», btc).
    currency: usd, eur, rub и другие коды. По умолчанию usd.
    """
    logger.info("crypto_price: %s / %s", coin, currency)
    try:
        price = get_crypto_price(coin, currency)
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    except requests.RequestException as exc:
        logger.warning("crypto error: %s", exc)
        return f"CoinGecko недоступен: {exc}"
    vs = _normalize_currency(currency).upper()
    coin_id = _normalize_coin(coin)
    return f"{coin_id}: {price} {vs} (источник: CoinGecko)"


_CBR_URL = "http://www.cbr.ru/scripts/XML_daily.asp"
_CBR_ALIASES = {
    "usd": "USD",
    "dollar": "USD",
    "доллар": "USD",
    "доллары": "USD",
    "бакс": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "евро": "EUR",
    "cny": "CNY",
    "юань": "CNY",
    "gbp": "GBP",
    "фунт": "GBP",
    "chf": "CHF",
    "франк": "CHF",
    "jpy": "JPY",
    "иена": "JPY",
    "йена": "JPY",
    "try": "TRY",
    "лира": "TRY",
    "kzt": "KZT",
    "тенге": "KZT",
    "byn": "BYN",
    "белорусский": "BYN",
    "uah": "UAH",
    "гривна": "UAH",
}


def _cbr_code(currency: str) -> str:
    key = (currency or "").strip().lower().replace(" ", "")
    if not key:
        raise ValueError("Не указана валюта.")
    code = _CBR_ALIASES.get(key, key.upper())
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError(f"Непонятная валюта: {currency}. Пример: USD, EUR, доллар.")
    return code


def _cbr_date(date: str) -> str:
    """Дата для параметра date_req ЦБ: ДД/ММ/ГГГГ."""
    raw = (date or "").strip()
    if not raw:
        return ""
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.strftime("%d/%m/%Y")
    raise ValueError("Дата должна быть в виде ДД.ММ.ГГГГ.")


def _cbr_number(value: str) -> float:
    return float((value or "0").replace(",", ".").replace(" ", ""))


def _fetch_cbr(date: str = "") -> tuple[str, list[dict[str, Any]]]:
    params: dict[str, str] = {}
    requested = _cbr_date(date)
    if requested:
        params["date_req"] = requested
    response = requests.get(
        _CBR_URL,
        params=params,
        headers={**HTTP_HEADERS, "Accept": "application/xml, text/xml"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows: list[dict[str, Any]] = []
    for node in root.findall("Valute"):
        nominal = int(_cbr_number(node.findtext("Nominal") or "1")) or 1
        value = _cbr_number(node.findtext("Value") or "0")
        rows.append(
            {
                "code": (node.findtext("CharCode") or "").upper(),
                "name": (node.findtext("Name") or "").strip(),
                "nominal": nominal,
                "value": value,
                "per_unit": value / nominal,
            }
        )
    if not rows:
        raise RuntimeError("ЦБ вернул пустой список курсов.")
    return root.attrib.get("Date", ""), rows


def get_cbr_rate(currency: str, date: str = "") -> float:
    """Сколько рублей стоит 1 единица валюты по курсу ЦБ. Возвращает число."""
    code = _cbr_code(currency)
    _, rows = _fetch_cbr(date)
    for row in rows:
        if row["code"] == code:
            return float(row["per_unit"])
    raise ValueError(f"В курсах ЦБ нет валюты {code}.")


def _format_cbr_row(row: dict[str, Any]) -> str:
    per_unit = float(row["per_unit"])
    nominal = int(row["nominal"])
    line = f"{row['code']}, {row['name']}: {per_unit:.4f} ₽ за 1"
    if nominal != 1:
        line += f" (официально {row['value']:.4f} ₽ за {nominal})"
    return line


@tool
def cbr_rate(currency: str = "", date: str = "") -> str:
    """Официальный курс Банка России: сколько рублей даёт ЦБ за единицу валюты.

    currency: USD, EUR, CNY или название («доллар», «евро», «юань»).
    Пустая строка — доллар и евро.
    date: необязательная дата ДД.ММ.ГГГГ. Пусто — последний опубликованный курс.
    Данные: http://www.cbr.ru/scripts/XML_daily.asp
    """
    logger.info("cbr_rate: %s / %s", currency or "USD,EUR", date or "latest")
    try:
        rate_date, rows = _fetch_cbr(date)
        if (currency or "").strip():
            code = _cbr_code(currency)
            chosen = [row for row in rows if row["code"] == code]
            if not chosen:
                return f"В курсах ЦБ нет валюты {code}."
        else:
            chosen = [row for row in rows if row["code"] in {"USD", "EUR"}]
    except ValueError as exc:
        return str(exc)
    except requests.RequestException as exc:
        logger.warning("cbr error: %s", exc)
        return f"Сервис ЦБ недоступен: {exc}"
    except (ET.ParseError, RuntimeError) as exc:
        logger.warning("cbr parse error: %s", exc)
        return f"Не удалось разобрать ответ ЦБ: {exc}"

    lines = [f"Дата курса ЦБ: {rate_date or 'не указана'}"]
    lines.extend(_format_cbr_row(row) for row in chosen)
    lines.append("Источник: Банк России, XML_daily.asp")
    return "\n".join(lines)


@tool
def run_shell(command: str) -> str:
    """Выполнить команду через PowerShell в папке files.

    Рабочие файлы команды попадают только в эту папку.
    Разрушительные команды (format, удаление дерева каталогов, выключение ПК) отклоняются.
    Команды, которые читают файлы секретов, тоже отклоняются.
    """
    command = (command or "").strip()
    if not command:
        return "Пустая команда."
    if len(command) > 500:
        return "Команда длиннее 500 символов — отклонена."
    if ".env" in command.lower() or "id_rsa" in command.lower():
        return "Отклонено: команда обращается к файлам с секретами."
    if re.search(r"(?i)(?:[a-z]:[\\/]|\\\\|~[\\/]|\.\.)", command):
        return "Отклонено: команда обращается к пути вне папки files."
    if _DANGEROUS_SHELL.search(command):
        logger.warning("shell blocked: %s", command)
        return "Отклонено: команда похожа на разрушительную и не выполнена."
    logger.info("run_shell: %s", command)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SHELL_TIMEOUT,
            cwd=str(FILES_DIR),
        )
    except subprocess.TimeoutExpired:
        return f"Команда прервана по таймауту {SHELL_TIMEOUT} с."
    except OSError as exc:
        return f"Не удалось запустить оболочку: {exc}"

    output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
    output = _clip(output.strip(), SHELL_MAX_CHARS) or "(команда ничего не вывела)"
    return f"Код выхода: {completed.returncode}\n{output}"


def _qr_squares(text: str) -> str:
    """Классический QR: тёмный модуль — два блока, чтобы в консоли он выглядел квадратом."""
    import qrcode

    code = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    code.add_data(text)
    code.make(fit=True)
    return "\n".join("".join("██" if cell else "  " for cell in row) for row in code.get_matrix())


@tool
def generate_qr(text: str) -> str:
    """Собрать QR-код и вернуть его рисунок из квадратов █. Файл не создаётся.

    text — строка или ссылка, которую нужно закодировать.
    Если текста нет, не выдумывай его.
    Рисунок из ответа нужно показать пользователю целиком.
    """
    text = (text or "").strip()
    if not text:
        return "Нужен текст или ссылка для QR-кода."
    if len(text) > 800:
        return "Текст длиннее 800 символов. Укоротите его."
    logger.info("generate_qr: %s символов", len(text))
    return _qr_squares(text)


def get_tools() -> list[Any]:
    return [
        web_search,
        http_request,
        read_file,
        write_file,
        list_directory,
        save_note,
        read_notes,
        get_weather,
        crypto_price,
        cbr_rate,
        run_shell,
        generate_qr,
    ]
