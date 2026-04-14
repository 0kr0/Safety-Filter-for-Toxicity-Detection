"""
Text normalization to defend against adversarial perturbations.
Reverses leetspeak, collapses repeated characters, removes inserted spaces,
and maps Cyrillic homoglyphs back to Latin equivalents.
"""

import re
import unicodedata

LEET_REVERSE = {
    "@": "a",
    "4": "a",
    "3": "e",
    "$": "s",
    "5": "s",
    "0": "o",
    "7": "t",
    "1": "i",
    "!": "i",
    "8": "b",
    "9": "g",
}

HOMOGLYPH_REVERSE = {
    "\u0430": "a",  # Cyrillic а
    "\u0441": "c",  # Cyrillic с
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0445": "x",  # Cyrillic х
    "\u0443": "y",  # Cyrillic у
    "\u041a": "K",  # Cyrillic К
    "\u041c": "M",  # Cyrillic М
    "\u0422": "T",  # Cyrillic Т
    "\u0410": "A",  # Cyrillic А
    "\u0412": "B",  # Cyrillic В
    "\u0415": "E",  # Cyrillic Е
    "\u041d": "H",  # Cyrillic Н
    "\u041e": "O",  # Cyrillic О
    "\u0420": "P",  # Cyrillic Р
    "\u0421": "C",  # Cyrillic С
    "\u0425": "X",  # Cyrillic Х
}


def reverse_leetspeak(text: str) -> str:
    """Replace common leetspeak characters with their Latin equivalents."""
    result = []
    for ch in text:
        result.append(LEET_REVERSE.get(ch, ch))
    return "".join(result)


def collapse_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Collapse runs of 3+ identical characters down to max_repeat."""
    if not text:
        return text
    result = [text[0]]
    count = 1
    for ch in text[1:]:
        if ch == result[-1]:
            count += 1
            if count <= max_repeat:
                result.append(ch)
        else:
            result.append(ch)
            count = 1
    return "".join(result)


def merge_spaced_words(text: str) -> str:
    """
    Detect sequences of single characters separated by spaces and merge them.
    E.g. 'i d i o t' -> 'idiot', but 'I am a person' stays unchanged.
    """
    words = text.split()
    if len(words) < 3:
        return text

    result = []
    i = 0
    while i < len(words):
        if len(words[i]) == 1 and words[i].isalpha():
            run = [words[i]]
            j = i + 1
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                run.append(words[j])
                j += 1
            if len(run) >= 3:
                result.append("".join(run))
            else:
                result.extend(run)
            i = j
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def reverse_homoglyphs(text: str) -> str:
    """Replace Cyrillic look-alike characters with Latin equivalents."""
    result = []
    for ch in text:
        result.append(HOMOGLYPH_REVERSE.get(ch, ch))
    return "".join(result)


def normalize_text(text: str) -> str:
    """
    Full normalization pipeline: homoglyphs -> leetspeak -> repeated chars -> spaced words.
    Order matters: homoglyphs first (Unicode), then leetspeak (ASCII substitutions),
    then structural fixes.
    """
    text = reverse_homoglyphs(text)
    text = reverse_leetspeak(text)
    text = collapse_repeated_chars(text)
    text = merge_spaced_words(text)
    return text
