from __future__ import annotations

import json
from pathlib import Path


INPUTS_PATH = Path("inputs/inputs.json")


def load_inputs(path: Path) -> list[dict]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of episodes.")
    return data


def find_next_diarization_episode(episodes: list[dict]) -> dict | None:
    for episode in episodes:
        if episode.get("status") == "diarization":
            return episode
    return None


def write_github_output(name: str, value: str) -> None:
    github_output = Path(
        __import__("os").environ.get("GITHUB_OUTPUT", "")
    )
    if github_output:
        with github_output.open("a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    episodes = load_inputs(INPUTS_PATH)
    episode = find_next_diarization_episode(episodes)

    should_run = "true" if episode else "false"
    episode_id = str(episode.get("id")) if episode else ""

    write_github_output("should_run", should_run)
    write_github_output("episode_id", episode_id)

    print(
        json.dumps(
            {
                "should_run": should_run == "true",
                "episode_id": episode_id or None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
