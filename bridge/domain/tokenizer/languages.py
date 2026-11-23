"""Which language each word of a row is in.

A row is one string. Its words are its maximal runs of non-whitespace, and every word
carries a language code. Both tokenizers and the dataset read a row through this module, so
they cannot disagree about where a word starts or which language it is in.
"""

import re
from collections.abc import Sequence

SUPPORTED_LANGUAGES = ("EN", "ES")
# The code of a row whose language was never given. It covers a whole row or none of it.
UNSPECIFIED = "--"

# One row's languages as a caller writes them: one code for every word (``"EN"``), one code
# per word (``"EN EN ES"``, or a list or tuple of codes), or ``None`` for unspecified.
type RowLanguages = str | Sequence[str | None] | None
# The languages of one tokenizer call: one entry per row, or ``None`` for no language at all.
type Languages = Sequence[RowLanguages] | None

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


def per_row_languages(texts: list[str], languages: Languages = None) -> list[tuple[str, ...]]:
    """The per-word languages of every row of one tokenizer call.

    ``texts`` is a list of rows. ``languages`` is a list or tuple holding one entry per row,
    each in a form :func:`per_word_languages` accepts, or ``None`` to leave every row
    unspecified::

        ["no me", "la con"],  ["ES", "EN"]            "no me" in ES, "la con" in EN
        ["no me", "la con"],  ["ES EN", None]         "no" in ES, "me" in EN; "la con" unspecified
        ["no me", "la con"],  [("ES", "EN"), "EN"]    the same first row, "la con" in EN

    One string is not read as a code for every row: it raises, since a row's own entry may
    be a string of several codes.
    """
    if not isinstance(texts, list) or not all(isinstance(row, str) for row in texts):
        raise TypeError("The rows to encode are a list of strings")
    if languages is None:
        languages = [None] * len(texts)
    elif not isinstance(languages, list | tuple):
        raise ValueError(
            f"Languages are a list or a tuple with one entry per row, got {languages!r}"
        )
    elif len(languages) != len(texts):
        raise ValueError(f"{len(languages)} language entries for {len(texts)} rows")
    return [per_word_languages(row, entry) for row, entry in zip(texts, languages, strict=True)]
