# SYSTEM IMPORTS
from collections.abc import Collection, Iterable, Mapping, MutableSet, Sequence, Set
from typing import Type, Tuple


# PYTHON PROJECT IMPORTS


# Types declared in this module
VocabType: Type = Type["Vocab"]


# Constants declared in this module
START_TOKEN: str = "<BOS>"
END_TOKEN: str = "<EOS>"
UNK_TOKEN: str = "<UNK>"


class Vocab(MutableSet):
    """Set-like data structure that can change words into numbers and back."""

    def __init__(self: VocabType) -> None:
        words: Set[str] = {START_TOKEN, END_TOKEN, UNK_TOKEN}
        self.num_to_word: Sequence[str] = list(words)    
        self.word_to_num: Mapping[str, int] = {word:num for num, word in enumerate(self.num_to_word)}

    def add(self: VocabType,
            word: str
            ) -> None:
        if word not in self:
            num: int = len(self.num_to_word)
            self.num_to_word.append(word)
            self.word_to_num[word] = num

    def discard(self: VocabType,
                word: str
                ) -> None:
        raise NotImplementedError()

    def update(self: VocabType,
               words: Collection[str]
               ) -> None:
        self |= words

    def __contains__(self: VocabType,
                     word: str
                     ) -> bool:
        return word in self.word_to_num

    def __len__(self: VocabType) -> int:
        return len(self.num_to_word)

    def __iter__(self: VocabType) -> Iterable[str]:
        return iter(self.num_to_word)

    def numberize(self: VocabType,
                  word: str
                  ) -> int:
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num[UNK_TOKEN]

    def denumberize(self: VocabType,
                    num: int
                    ) -> str:
        """Convert a number into a word."""
        return self.num_to_word[num]


def progress(iterable: Iterable) -> Iterable:
    """Iterate over `iterable`, showing progress if appropriate."""
    try:
        import tqdm
        return tqdm.tqdm(iterable, disable=None)
    except ImportError:
        return iterable


def split(line: str,
          delim: str = None
          ) -> Sequence[str]:
    line: str = line.rstrip('\r\n')
    if delim == '':
        return list(line)
    else:
        return line.split(delim)

def read_parallel(ffilename: str,
                  efilename: str,
                  delim: str = None
                  ) -> Sequence[Tuple[str, str]]:
    """Read data from the files named by `ffilename` and `efilename`.

    The files should have the same number of lines.

    Arguments:
      - ffilename: str
      - efilename: str
      - delim: delimiter between symbols (default: any whitespace)
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    data: Sequence[Tuple[str, str]] = []
    for (fline, eline) in zip(open(ffilename), open(efilename)):
        fwords = [START_TOKEN] + split(fline, delim) + [END_TOKEN]
        ewords = [START_TOKEN] + split(eline, delim) + [END_TOKEN]
        data.append((fwords, ewords))
    return data

def read_mono(filename: str,
              delim: str = None
              ) -> Sequence[str]:
    """Read sentences from the file named by `filename`.

    Arguments:
      - filename
      - delim: delimiter between symbols (default: any whitespace)
    Returns: list of lists of strings. <BOS> and <EOS> are added to each sentence.
    """
    data: Sequence[str] = []
    for line in open(filename):
        words = [START_TOKEN] + split(line, delim) + [END_TOKEN]
        data.append(words)
    return data

def write_mono(data: Sequence[str],
               filename: str,
               delim: str = ' '
               ) -> None:
    """Write sentences to the file named by `filename`.

    Arguments:
      - data: list of lists of strings. <BOS> and <EOS> are stripped off.
      - filename: str
      - delim: delimiter between symbols (default: space)
    """
    with open(filename, 'w') as outfile:
        for words in data:
            if len(words) > 0 and words[0] == START_TOKEN: words.pop(0)
            if len(words) > 0 and words[-1] == END_TOKEN: words.pop(-1)
            print(delim.join(words), file=outfile)

