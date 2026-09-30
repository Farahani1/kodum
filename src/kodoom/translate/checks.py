"""Automatic checks on a translation (plan 1.2, step 3).

They run on every case and catch what a reviewer should not have to: a changed
identifier, a lost number, output that is not Persian, empty or looping. They do not
judge meaning; that is the checker model's and the reviewers' job. A check returns
findings; an empty list means the text passed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from kodoom.normalize import to_latin_digits
from kodoom.translate.rules import KEEP, RULES, Segment, segments

MIN_PERSIAN_SHARE = 0.6
MIN_LENGTH_RATIO = 0.3
MAX_LENGTH_RATIO = 3.0
RATIO_FROM_CHARS = 20  # shorter sources have no meaningful length ratio
MIN_LETTERS_FOR_SCRIPT = 6

_BACKTICKED = re.compile(r"`[^`]+`")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://\S+")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.:/][A-Za-z0-9]+)+|[A-Za-z]+\d+[A-Za-z0-9]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_PERSIAN_LETTER = re.compile("[ء-غف-يپچژکگیۀ]")
_LATIN_LETTER = re.compile("[A-Za-z]")
_WORD = re.compile(r"\w+")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


@dataclass(frozen=True)
class Finding:
    check: str
    where: str
    message: str


def protected_tokens(text: str) -> list[str]:
    """Spans that must reach the translation unchanged: code in backticks, emails,
    URLs and identifiers (a letter-led token with a digit, ``_``, ``:``, ``/`` or a
    ``-``/``.`` joining alphanumerics that contains a digit)."""
    found = []
    rest = text
    for pattern in (_BACKTICKED, _URL, _EMAIL):
        found += pattern.findall(rest)
        rest = pattern.sub(" ", rest)
    for token in _IDENTIFIER.findall(rest):
        if any(c.isdigit() or c in "_:/" for c in token):
            found.append(token)
    return found


def numbers(text: str) -> list[str]:
    """Numbers as Latin digits with ASCII separators, so ۱۲٫۵ equals 12.5."""
    text = to_latin_digits(text).replace("٫", ".").replace("٬", ",")
    return _NUMBER.findall(text)


def _without_protected(text: str) -> str:
    for pattern in (_BACKTICKED, _URL, _EMAIL, _IDENTIFIER):
        text = pattern.sub(" ", text)
    return text


def untranslated_words(source: str, target: str, keep: Sequence[str] = ()) -> list[str]:
    """English words of the source that are still in the translation.

    Only words that are common words in the source count (lower case, or first in a
    sentence); capitalized words inside a sentence are names and may stay. Code,
    identifiers, emails, URLs and the keep-in-English terms are not counted.
    """

    def plain(text: str) -> str:
        text = _without_protected(text)
        for term in keep:
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", " ", text)
        return text

    source_body, target_body = plain(source), plain(target)
    common = set()
    for m in _LATIN_WORD.finditer(source_body):
        before = source_body[: m.start()].rstrip()
        if m.group()[0].islower() or not before or before[-1] in ".!?:;\n":
            common.add(m.group().lower())
    return sorted({w.lower() for w in _LATIN_WORD.findall(target_body)} & common)


def check_text(
    source: str, target: str, where: str = "", keep: Sequence[str] = ()
) -> list[Finding]:
    """The checks for one translated text. ``keep``: terms that stay in English."""
    findings: list[Finding] = []

    def add(check: str, message: str) -> None:
        findings.append(Finding(check, where, message))

    if not target.strip():
        return [Finding("empty", where, "the translation is empty")]

    missing = [t for t in protected_tokens(source) if t not in target]
    if missing:
        add("identifiers", f"not carried over unchanged: {missing}")

    lost = Counter(numbers(source)) - Counter(numbers(target))
    if lost:
        add("numbers", f"numbers lost or changed: {sorted(lost.elements())}")

    foreign = foreign_letters(target)
    if foreign:
        add("script", f"letters from another script: {foreign}")

    body = _without_protected(target)
    persian, latin = len(_PERSIAN_LETTER.findall(body)), len(_LATIN_LETTER.findall(body))
    if (
        persian + latin >= MIN_LETTERS_FOR_SCRIPT
        and persian / (persian + latin) < MIN_PERSIAN_SHARE
    ):
        add("script", f"only {persian} of {persian + latin} letters are Persian")

    if len(source) >= RATIO_FROM_CHARS:
        ratio = len(target) / len(source)
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            add(
                "length",
                f"length ratio {ratio:.2f} is outside {MIN_LENGTH_RATIO}-{MAX_LENGTH_RATIO}",
            )

    left = untranslated_words(source, target, keep)
    if left:
        add("english", f"English words left in the translation: {left}")

    loop = repeated_phrase(target)
    if loop:
        add("looping", f"repeats {loop!r}")
    return findings


def foreign_letters(text: str) -> list[str]:
    """Letters that are neither Persian/Arabic-script nor Latin, with their script.

    A model sometimes slips a look-alike letter of another alphabet into a word (a Cyrillic
    «о» in «مونیتورینگ»); the word then looks right to a reader and fails every string
    comparison.
    """
    found = []
    for ch in dict.fromkeys(text):
        if ch.isalpha():
            script = unicodedata.name(ch, "UNKNOWN").split()[0]
            if script not in ("ARABIC", "LATIN"):
                found.append(f"{ch} ({script.lower()})")
    return found


def repeated_phrase(text: str, times: int = 4) -> str | None:
    """A word or phrase (up to 6 words) that repeats back to back ``times`` or more times."""
    words = _WORD.findall(text)
    for n in range(1, 7):
        for start in range(len(words) - n * times + 1):
            phrase = words[start : start + n]
            if all(words[start + k * n : start + (k + 1) * n] == phrase for k in range(1, times)):
                return " ".join(phrase)
    return None


_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z \-]{0,30}?)\s*:")


def check_option_labels(
    source: Sequence[str], target: Sequence[str], where: str = ""
) -> list[Finding]:
    """Score levels start with a label ("Low: ...", "High: ..."); the labels of the
    translation must exist and differ, or the levels stop being distinguishable.

    Trial 6: a model turned «Low», «Moderate» and «High» into «سطح», «شدت» and «شدت».
    """
    if len(source) != len(target) or sum(bool(_LABEL.match(s)) for s in source) < 2:
        return []
    labels = []
    for s, t in zip(source, target, strict=True):
        if not _LABEL.match(s):
            continue
        head = re.split(r"[:\uff1a]", t, maxsplit=1)
        labels.append(head[0].strip() if len(head) == 2 else "")
    if "" in labels:
        return [Finding("labels", where, "an option lost its label before the colon")]
    if len(set(labels)) < len(labels):
        return [Finding("labels", where, f"option labels are not distinct: {labels}")]
    return []


def check_state(workflow: str, source: str, target: str, keep: Sequence[str] = ()) -> list[Finding]:
    """Structure, kept fields and translated fields of a whole state."""
    if workflow not in RULES:
        raise KeyError(workflow)
    try:
        t_tree = json.loads(target)
    except json.JSONDecodeError as e:
        return [Finding("structure", "state", f"translated state is not valid JSON: {e}")]
    s_tree = json.loads(source)
    findings: list[Finding] = []
    if _shape(s_tree) != _shape(t_tree):
        return [Finding("structure", "state", "keys or list lengths differ from the source")]
    for seg in segments(workflow, source):
        translated = _at(t_tree, seg.location)
        where = "state." + seg.path
        if seg.rule.action == KEEP:
            if translated != seg.text:
                findings.append(Finding("keep", where, f"{seg.text!r} became {translated!r}"))
        elif not isinstance(translated, str):
            findings.append(Finding("structure", where, "is no longer text"))
        else:
            findings += check_text(seg.text, translated, where, keep)
    findings += _other_leaves(s_tree, t_tree)
    return findings


def _shape(node):
    if isinstance(node, dict):
        return {k: _shape(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_shape(v) for v in node]
    return type(node).__name__ if not isinstance(node, str) else "str"


def _at(tree, location):
    for step in location:
        tree = tree[step]
    return tree


def _other_leaves(source, target, path="state") -> list[Finding]:
    """Non-text leaves (numbers, booleans, null) must be identical."""
    if isinstance(source, dict):
        return [f for k in source for f in _other_leaves(source[k], target[k], f"{path}.{k}")]
    if isinstance(source, list):
        return [f for i, v in enumerate(source) for f in _other_leaves(v, target[i], f"{path}[]")]
    if not isinstance(source, str) and source != target:
        return [Finding("keep", path, f"{source!r} became {target!r}")]
    return []


def translated_segments(workflow: str, state: str) -> list[Segment]:
    """The segments a translator has to translate."""
    return [s for s in segments(workflow, state) if s.rule.action != KEEP]
