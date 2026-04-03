from .schemas import Utterance


def _format_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_srt(path, utterances: list[Utterance]):
    lines = []

    for i, u in enumerate(utterances, 1):
        lines.append(str(i))
        lines.append(f"{_format_time(u.start)} --> {_format_time(u.end)}")
        lines.append(f"[{u.speaker}] {u.text}")
        lines.append("")

    path.write_text("\n".join(lines))
