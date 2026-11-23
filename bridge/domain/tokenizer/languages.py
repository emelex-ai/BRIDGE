"""Which language each word of a row is in.

A row is one string. Its words are its maximal runs of non-whitespace, and every word
carries a language code. Both tokenizers and the dataset read a row through this module, so
they cannot disagree about where a word starts or which language it is in.
"""

import re
from collections.abc import Sequence
from typing import cast

SUPPORTED_LANGUAGES = ("EN", "ES")
# The code of a row whose language was never given. It covers a whole row or none of it.
UNSPECIFIED = "--"

# One row's languages as a caller writes them: one code for every word (``"EN"``), one code
# per word (``"EN EN ES"``, or a list or tuple of codes), or ``None`` for unspecified.
type RowLanguages = str | Sequence[str | None] | None
# The languages of one ``encode`` call: see :func:`rows_and_languages`.
type Languages = RowLanguages | Sequence[RowLanguages]

_WORD = re.compile(r"\S+")


def split_words(text: str) -> list[re.Match[str]]:
    """The words of ``text``, in order: each match is one word and knows its offset."""
    return list(_WORD.finditer(text))


def per_word_languages(text: str, languages: RowLanguages = None) -> tuple[str, ...]:
    """One upper-case language code per word of ``text``.

    A single code covers every word. ``None`` and ``"--"`` both mean unspecified, for the
    row or as a word's code, and unspecified is a property of the whole row: beside a real
    code it raises. A row with no words still opens with a language token, so it takes one
    code and returns one.

    Raises ``ValueError`` for anything but a string, a list or a tuple, which are the forms
    whose order is the word order; for no code at all; for an unsupported code; and for a
    count that is neither one nor the number of words.
    """
    n_words = len(split_words(text))
    if languages is None:
        return (UNSPECIFIED,) * max(n_words, 1)
    if not isinstance(languages, str | list | tuple):
        raise ValueError(
            f"Languages are a string, a list or a tuple of codes, got {languages!r} for {text!r}"
        )

    codes = (
        languages.split()
        if isinstance(languages, str)
        else [UNSPECIFIED if code is None else code for code in languages]
    )
    if not codes:
        raise ValueError(f"No language code given for {text!r}")
    if not all(isinstance(code, str) for code in codes):
        raise ValueError(f"A language code is a string, got {languages!r} for {text!r}")
    codes = [code.upper() for code in codes]
    invalid = sorted(set(codes) - {*SUPPORTED_LANGUAGES, UNSPECIFIED})
    if invalid:
        raise ValueError(
            f"Invalid languages: {invalid}. Supported languages are: {list(SUPPORTED_LANGUAGES)}"
        )
    if UNSPECIFIED in codes and set(codes) != {UNSPECIFIED}:
        raise ValueError(
            f"{UNSPECIFIED!r} means the whole row is unspecified and cannot sit beside a "
            f"language: {codes} for {text!r}"
        )
    if len(codes) == 1:
        return tuple(codes) * max(n_words, 1)
    if len(codes) != n_words:
        raise ValueError(f"{len(codes)} language codes for {n_words} words in {text!r}: {codes}")
    return tuple(codes)


def rows_and_languages(
    text: str | list[str], languages: Languages = None
) -> tuple[list[str], list[tuple[str, ...]]]:
    """The rows of one ``encode`` call and each row's per-word languages.

    ``text`` is one row or a list of rows. For one row, ``languages`` is that row's
    languages. For a list, ``languages`` is ``None``, one string applied to every row, or a
    list or tuple holding one entry per row. So a list of codes is read per word beside one
    row and per row beside a list of rows::

        "no me",              ["ES", "EN"]    one row: "no" in ES, "me" in EN
        ["no me", "la con"],  ["ES", "EN"]    two rows: "no me" in ES, "la con" in EN
        ["no me", "la con"],  "ES EN"         two rows, each ES then EN
    """
    if isinstance(text, str):
        return [text], [per_word_languages(text, cast(RowLanguages, languages))]
    if not isinstance(text, list) or not all(isinstance(row, str) for row in text):
        raise TypeError("Input must be a string or a list of strings")
    if languages is None or isinstance(languages, str):
        per_row: Sequence[RowLanguages] = [languages] * len(text)
    elif not isinstance(languages, list | tuple):
        raise ValueError(
            f"Languages for a list of rows are a string, a list or a tuple, got {languages!r}"
        )
    elif len(languages) != len(text):
        raise ValueError(f"{len(languages)} language entries for {len(text)} rows")
    else:
        per_row = languages
    return text, [per_word_languages(row, spec) for row, spec in zip(text, per_row, strict=True)]
