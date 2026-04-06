from __future__ import annotations

from .schemas import IdentityDecision, IdentityQAIssue, IdentityQAResult


def run_identity_qa(
    expected_speakers: set[str],
    decisions: list[IdentityDecision],
) -> IdentityQAResult:
    issues: list[IdentityQAIssue] = []
    checks: dict[str, bool] = {}

    decision_speakers = [item.speaker for item in decisions]
    decision_speaker_set = set(decision_speakers)

    checks["all_expected_speakers_present"] = expected_speakers == decision_speaker_set
    if not checks["all_expected_speakers_present"]:
        issues.append(
            IdentityQAIssue(
                code="speaker_set_mismatch",
                message=(
                    f"Expected speakers {sorted(expected_speakers)} but got "
                    f"{sorted(decision_speaker_set)}"
                ),
            )
        )

    checks["no_duplicate_speakers"] = len(decision_speakers) == len(decision_speaker_set)
    if not checks["no_duplicate_speakers"]:
        issues.append(
            IdentityQAIssue(
                code="duplicate_speakers",
                message="Duplicate speakers found in final decisions.",
            )
        )

    checks["confidence_in_range"] = all(0.0 <= item.confidence <= 1.0 for item in decisions)
    if not checks["confidence_in_range"]:
        issues.append(
            IdentityQAIssue(
                code="confidence_out_of_range",
                message="At least one confidence value is outside [0.0, 1.0].",
            )
        )

    checks["display_name_present"] = all(bool(item.proposed_display_name.strip()) for item in decisions)
    if not checks["display_name_present"]:
        issues.append(
            IdentityQAIssue(
                code="missing_display_name",
                message="At least one final decision has an empty display name.",
            )
        )

    passed = not issues
    return IdentityQAResult(
        passed=passed,
        checks=checks,
        issues=issues,
    )
