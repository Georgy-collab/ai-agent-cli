"""Агент: выбирает инструмент LangChain и отвечает через ProxyAPI (gpt-4o-mini)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.tools import append_conversation, ensure_storage, get_tools, memory_context

logger = logging.getLogger("agent")

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.proxyapi.ru/openai/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_STEPS = 8

SYSTEM_PROMPT = """Ты локальный помощник в терминале. Отвечай на русском языке, коротко и по делу.
Сам выбирай инструмент под задачу:
- свежая информация из интернета, новости, справка → web_search
- погода, температура, ветер → get_weather (передавай название города)
- официальный курс валюты к рублю по ЦБ → cbr_rate (код или название: USD, EUR, доллар, евро). Это не криптовалюта
- цена Bitcoin, Ethereum и других монет → crypto_price
- вызов произвольного HTTP API → http_request
- прочитать, записать или показать файлы → read_file, write_file, list_directory
  Рабочие файлы только в папке files в корне проекта. Не читай и не записывай ничего вне неё.
  В путь передавай имя файла, например notes.txt, без корня проекта.
- выполнить локальное действие → run_shell
  Команда запускается в папке files, файлы команды сохраняются туда же.
- запомнить факт между сессиями → save_note
- вспомнить сохранённое → read_notes
- QR-код по тексту или ссылке → generate_qr
  Вызывай инструмент при каждом таком запросе, даже если похожий код уже был.
  Рисунок в ответ вставляй целиком, символ в символ, и только потом одну короткую фразу.

Если не хватает данных (город, монета, валюта, путь, что именно сделать), не выдумывай значения.
Задай один короткий уточняющий вопрос.
После инструментов дай структурированный ответ: суть, затем детали (числа, единицы, источник).
Не упоминай внутренние имена функций, если пользователь об этом не просил.
Команды оболочки запускай только когда пользователь просит действие с файлами в папке files.
"""


def setup_logging() -> None:
    """Пишет лог только в agent/logs/agent.log. В консоль сообщения не выводятся."""
    log = logging.getLogger("agent")
    if log.handlers:
        return
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    log_dir = AGENT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # Сторонние библиотеки не должны писать предупреждения в терминал.
    for name in ("httpx", "httpcore", "openai", "urllib3", "ddgs", "primp", "langchain", "langchain_openai"):
        third_party = logging.getLogger(name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False


def load_environment() -> None:
    """Читает .env из корня проекта и, если есть, из agent/.env. Ключи не логируются."""
    load_dotenv(ROOT / ".env")
    load_dotenv(AGENT_DIR / ".env", override=True)


def _api_key() -> str:
    for name in ("PROXYAPI_API_KEY", "OPENAI_API_KEY"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise RuntimeError(
        "Не найден ключ. Добавьте PROXYAPI_API_KEY в файл .env в корне проекта."
    )


def build_llm() -> ChatOpenAI:
    load_environment()
    model = (os.getenv("PROXYAPI_MODEL") or DEFAULT_MODEL).strip()
    base_url = (os.getenv("PROXYAPI_BASE_URL") or DEFAULT_BASE_URL).strip()
    logger.info("Модель %s через %s", model, base_url)
    return ChatOpenAI(
        model=model,
        api_key=_api_key(),
        base_url=base_url,
        temperature=0.2,
        max_retries=2,
    )


class LocalAgent:
    """Цикл tool calling: модель сама решает, какой инструмент вызвать."""

    def __init__(self) -> None:
        ensure_storage()
        self.tools = get_tools()
        self.tool_map = {item.name: item for item in self.tools}
        self.llm = build_llm().bind_tools(self.tools)
        self.history: list = []

    def _system_message(self) -> SystemMessage:
        remembered = memory_context()
        text = SYSTEM_PROMPT
        if remembered:
            text += "\nДолговременная память:\n" + remembered
        return SystemMessage(content=text)

    def ask(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return "Напишите задачу."

        logger.info("Запрос: %s", user_text[:300])
        messages: list = [self._system_message(), *self.history, HumanMessage(content=user_text)]
        tools_used: list[str] = []
        qr_picture = ""
        final = ""

        for step in range(1, MAX_TOOL_STEPS + 1):
            ai_message = self.llm.invoke(messages)
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(content=str(ai_message))
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                final = _message_text(ai_message)
                break

            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args") or {}
                call_id = call.get("id") or f"call_{step}_{name}"
                tools_used.append(name)
                logger.info("Шаг %s: инструмент %s, аргументы %s", step, name, _safe_args(args))
                observation = _run_tool(self.tool_map, name, args)
                if name == "generate_qr" and "█" in observation:
                    qr_picture = observation
                logger.debug("Результат %s: %s", name, observation[:500])
                messages.append(ToolMessage(content=observation, tool_call_id=call_id))
        else:
            final = "Не удалось закончить задачу за отведённое число шагов. Уточните запрос."

        if not final:
            final = "Пустой ответ модели. Попробуйте переформулировать вопрос."
        if _wants_qr(user_text) and not qr_picture:
            qr_picture = _run_tool(self.tool_map, "generate_qr", {"text": _qr_text(user_text)})
            if "generate_qr" not in tools_used:
                tools_used.append("generate_qr")
        if qr_picture and "█" not in final:
            final = qr_picture

        self.history.append(HumanMessage(content=user_text))
        self.history.append(AIMessage(content=final))
        # В сессии держим только реплики, без сырых вызовов инструментов.
        self.history = self.history[-12:]

        try:
            append_conversation(user_text, final, tools_used)
        except OSError as exc:
            logger.warning("Не удалось записать memory.json: %s", exc)

        return final


def _wants_qr(text: str) -> bool:
    lowered = text.lower()
    return "qr" in lowered or "кьюар" in lowered or "куар" in lowered


def _qr_text(text: str) -> str:
    quoted = re.findall(r"[\"«'](.+?)[\"»']", text)
    if quoted:
        return quoted[-1].strip()
    match = re.search(r"(?:информаци(?:ей|ю)|для|с текстом)\s+(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" .")
    return text.strip()


def _run_tool(tool_map: dict, name: str, args: object) -> str:
    tool = tool_map.get(name)
    if tool is None:
        return f"Инструмент «{name}» недоступен."
    if not isinstance(args, dict):
        args = {}
    try:
        result = tool.invoke(args)
    except Exception as exc:  # инструмент не должен ронять весь диалог
        logger.exception("Инструмент %s завершился с ошибкой", name)
        return f"Инструмент {name} завершился с ошибкой: {exc}"
    return result if isinstance(result, str) else str(result)


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _safe_args(args: object) -> object:
    """Укорачивает аргументы в логе, чтобы не писать большие тела файлов."""
    if not isinstance(args, dict):
        return args
    compact = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 180:
            compact[key] = value[:180] + "…"
        else:
            compact[key] = value
    return compact
