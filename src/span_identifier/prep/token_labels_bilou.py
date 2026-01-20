from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str = "internal"


def make_label_map(entity_type: str = "internal") -> Dict[str, int]:
    """
    BILOU tag set for a single entity type.
    Output includes:
      O, B-<T>, I-<T>, L-<T>, U-<T>
    """
    t = entity_type.upper()
    labels = ["O", f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"]
    return {lab: i for i, lab in enumerate(labels)}


def spans_to_bilou_for_tokens(
    offsets: List[Tuple[int, int]],
    spans: List[Span],
    label_map: Dict[str, int],
    ignore_label_id: int = -100,
) -> List[int]:
    """
    Convert character-level spans to token-level BILOU labels using token offsets.

    Args:
        offsets: list of (start_char, end_char) per token from a fast tokenizer.
        spans: list of gold spans in character offsets, relative to the same text.
        label_map: label -> id mapping (must include O, B-*, I-*, L-*, U-*).
        ignore_label_id: id to use for tokens you want ignored (caller may override further).

    Returns:
        labels: list[int] of size len(offsets)
    """
    # Default: O everywhere
    labels = [label_map.get("O", 0)] * len(offsets)

    if not spans:
        return labels

    # Sort spans by start to make behavior deterministic
    spans_sorted = sorted(spans, key=lambda s: (s.start, s.end))

    for sp in spans_sorted:
        if sp.end <= sp.start:
            continue

        # Determine token indices that overlap span
        token_idxs: List[int] = []
        for ti, (ts, te) in enumerate(offsets):
            # Many special tokens have (0,0); caller masks them anyway,
            # but we avoid labeling them.
            if te <= ts:
                continue

            # Overlap condition: [ts,te) intersects [sp.start, sp.end)
            if te <= sp.start or ts >= sp.end:
                continue
            token_idxs.append(ti)

        if not token_idxs:
            continue

        t = sp.type.upper()
        B = f"B-{t}"
        I = f"I-{t}"
        L = f"L-{t}"
        U = f"U-{t}"

        if len(token_idxs) == 1:
            labels[token_idxs[0]] = label_map.get(U, label_map.get("O", 0))
        else:
            labels[token_idxs[0]] = label_map.get(B, label_map.get("O", 0))
            for mid in token_idxs[1:-1]:
                labels[mid] = label_map.get(I, label_map.get("O", 0))
            labels[token_idxs[-1]] = label_map.get(L, label_map.get("O", 0))

    return labels
