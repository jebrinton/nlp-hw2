# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
import os
import sys


# PYTHON PROJECT IMPORTS


def load_aligned_data(fpath: str,
                      delimiter: str = "\t"
                      ) -> Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]:
    first_lang = list()
    second_lang = list()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.split(delimiter)
            if len(split_line) != 2:
                raise Exception("line [%s] split on delimiter [%s] has length [%s]" %
                    (line, delimiter, len(split_line)))
            lang_one, lang_two = split_line
            first_lang.append(lang_one.split())
            second_lang.append(lang_two.split())
    return first_lang, second_lang

def load_unaligned_data(fpath: str) -> Sequence[Sequence[str]]:
    data = list()
    with open(fpath, "r") as f:
        for line in f:
            data.append(line.split())
    return data

