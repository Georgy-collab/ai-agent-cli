"""CLI: python run.py  или  python agent/run.py  [вопрос]."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import LocalAgent, setup_logging  # noqa: E402


HELP = """Команды:
  /help     эта справка
  /memory   показать, что лежит в memory.json
  /clear    забыть реплики текущей сессии (файл памяти не стирается)
  /exit     выйти

Примеры:
  Какая погода в Казани?
  Сколько стоит Bitcoin в рублях?
  Какой курс доллара по ЦБ?
  Найди, что такое ProxyAPI
  Сделай QR-код для https://example.com
"""


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _print_memory() -> None:
    from agent.tools import load_memory

    data = load_memory()
    notes = data.get("notes") or []
    conversations = data.get("conversations") or []
    print(f"Заметок: {len(notes)}, диалогов: {len(conversations)}")
    for note in notes[-5:]:
        print(f"  заметка: {note.get('text', '')}")
    for item in conversations[-5:]:
        print(f"  {item.get('summary', '')}")


def main() -> None:
    _configure_stdio()
    setup_logging()

    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(HELP)
        return

    try:
        agent = LocalAgent()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        print(agent.ask(question))
        return

    print("Локальный агент (gpt-4o-mini через ProxyAPI).")
    print("Напишите задачу на обычном языке. Выход: /exit")
    print(HELP)

    while True:
        try:
            user_text = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in {"/exit", "/quit", "exit", "quit", "выход"}:
            print("Выход.")
            break
        if lowered == "/help":
            print(HELP)
            continue
        if lowered == "/memory":
            _print_memory()
            continue
        if lowered == "/clear":
            agent.history.clear()
            print("История этой сессии очищена.")
            continue
        try:
            answer = agent.ask(user_text)
        except KeyboardInterrupt:
            print("\nОстановлено.")
            continue
        except Exception as exc:  # сеть, ключ, неожиданный ответ API
            print(f"Ошибка: {exc}")
            continue
        print(f"\nАгент> {answer}")


if __name__ == "__main__":
    main()
