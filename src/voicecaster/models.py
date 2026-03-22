from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl

EpisodeStatus = Literal["pending", "processing", "pending_review", "done", "failed", "incompatible"]


class EpisodeEntry(BaseModel):
    id: str = Field(min_length=1)
    podcast_title: str = Field(min_length=1)
    episode_title: str = Field(min_length=1)
    url: HttpUrl
    topics: Optional[str] = None
    participants: Optional[list[str]] = None
    status: EpisodeStatus
    retries: int = Field(ge=0, default=0)


class AuditReview(BaseModel):
    identity_review_done: bool = False
    srt_audit_done: bool = False
    approved_as_source_of_truth: bool = False
    speaker_mapping_final: dict[str, str] = Field(default_factory=dict)
