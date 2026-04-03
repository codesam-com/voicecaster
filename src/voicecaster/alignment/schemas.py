@dataclass(slots=True)
class SubtitleCue:
    cue_id: str
    utterance_id: str
    source_segment_ids: list[int]
    start: float
    end: float
    speaker: str
    text: str
    line_count: int
    char_count: int
    word_count: int
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)
