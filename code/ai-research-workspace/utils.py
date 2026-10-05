"""Small helper functions used by the Streamlit UI."""

from __future__ import annotations

from datetime import datetime


def format_file_size(num_bytes: int) -> str:
    """Convert a byte count to a readable string, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - loop always returns


def chat_history_to_markdown(history: list[dict]) -> str:
    """Export the chat history (with sources) as a markdown document."""
    lines = [
        "# AI Research Workspace - Chat Export",
        "",
        f"_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for message in history:
        speaker = "You" if message.get("role") == "user" else "Assistant"
        lines.append(f"### {speaker}")
        lines.append("")
        lines.append(str(message.get("content", "")))
        lines.append("")

        sources = message.get("sources") or []
        if sources:
            lines.append("**Sources:**")
            for source in sources:
                lines.append(
                    f"- [{source.get('index', '?')}] {source.get('filename', 'unknown')}"
                    f", page {source.get('page', '?')}"
                )
            lines.append("")
    return "\n".join(lines)