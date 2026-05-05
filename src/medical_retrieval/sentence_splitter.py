from __future__ import annotations


_CLOSING_PUNCTUATION = frozenset(")]}\"'’”")


def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            _append_span(spans, text, start, i)
            start = i + 1
            i += 1
            continue
        if char in ".!?" and _is_sentence_boundary(text, i):
            end = _include_closing_punctuation(text, i + 1)
            _append_span(spans, text, start, end)
            start = end
            i = end
            continue
        i += 1

    _append_span(spans, text, start, len(text))
    return spans


def sentence_texts(text: str) -> list[str]:
    return [span[0] for span in sentence_spans(text)]


def _append_span(spans: list[tuple[str, int, int]], text: str, start: int, end: int) -> None:
    raw = text[start:end]
    sentence = raw.strip()
    if not sentence:
        return
    leading = len(raw) - len(raw.lstrip())
    span_start = start + leading
    spans.append((sentence, span_start, span_start + len(sentence)))


def _is_sentence_boundary(text: str, index: int) -> bool:
    if _is_decimal_point(text, index):
        return False
    lookahead = _include_closing_punctuation(text, index + 1)
    return lookahead >= len(text) or text[lookahead].isspace()


def _is_decimal_point(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _include_closing_punctuation(text: str, index: int) -> int:
    while index < len(text) and text[index] in _CLOSING_PUNCTUATION:
        index += 1
    return index


__all__ = ["sentence_spans", "sentence_texts"]
