# SYSTEM IMPORTS
from collections.abc import Sequence
import multiprocessing
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from fst import Transition
from from_file import load_aligned_data, load_unaligned_data
from utils import *
from models.ibm1 import IBM1
from models.translator import Translator
from eval.bleu import bleu


def parta(mandarin_corpus: Sequence[Sequence[str]],
          english_corpus: Sequence[Sequence[str]],
          test_corpus: Sequence[Sequence[str]]
          ) -> Translator:

    # need to compute the unique set of tokens in test_corpus
    # so we can add transitions that delete these tokens to our model
    # (just in case they're unknown tokens....we're choosing to delete unknown tokens instead of deal with
    #  them properly)
    unique_test_token_set = set()
    for seq in test_corpus:
        unique_test_token_set.update(seq)

    # threshold for EM to converge (make it small)
    error = 1e-10

    return Translator().train_from_raw(mandarin_corpus,
                                       english_corpus,
                                       unknown_fs=unique_test_token_set,
                                       converge_threshold=error)


def partb(m: Translator,
          test_corpus: Sequence[Sequence[str]],
          out_path: str,
          gold_path: str
          ) -> float:

    # print out the first 10 translations
    print_limit = 10
    with open(out_path, "w") as f:
        for i, (decoded_seq, _) in enumerate(m.decode(test_corpus)):
            if i < print_limit:
                print(decoded_seq)

            # make sure to write every translation to an output file so we can include it in our submission
            f.write(decoded_seq + "\n")

    # calculate (bleu will print the score) and print the bleu score
    return bleu(out_path, gold_path)


if __name__ == "__main__":
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    train_path = os.path.join(data_dir, "train.zh-en")
    test_path = os.path.join(data_dir, "test.zh")
    gold_path = os.path.join(data_dir, "test.en")

    translation_out = os.path.join(generated_dir, "part2_translations_new.out")

    mandarin_corpus, english_corpus = load_aligned_data(train_path)
    test_corpus = load_unaligned_data(test_path)

    m = parta(mandarin_corpus, english_corpus, test_corpus)

    partb(m, test_corpus, translation_out, gold_path)

