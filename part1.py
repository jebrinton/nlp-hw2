# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from typing import Tuple
import numpy as np
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from fst import Transition
from from_file import load_aligned_data
from utils import *
from models.ibm1 import IBM1
from eval.align_f1 import align_f1


def parta(train_path: str) -> Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]:
    return load_aligned_data(train_path, delimiter="\t")


def partb(mandarin_corpus: Sequence[Sequence[str]],
          english_corpus: Sequence[Sequence[str]]
          ) -> IBM1:
    m = IBM1()
    em_lls: np.ndarray = np.array(m.train_from_raw(mandarin_corpus,
                                                   english_corpus,
                                                   max_iter=100))
    em_lls_iter_1_and_on = em_lls[1:]
    em_lls_except_last_iter = em_lls[:-1]

    if(np.all(em_lls_iter_1_and_on >= em_lls_except_last_iter)):
        print("log-likelihoods monotonically increased! Good!")
    else:
        raise ValueError("[ERROR]: log-likelihoods did not monotonically increase...something is wrong with your EM!")

    # print out the probabilities for the following pairs
    pair_tuples = [
        tuple(["jedi", "绝地"]),
        tuple(["droid", "机械人"]),
        tuple(["force", "原力"]),
        tuple(["midi-chlorians", "原虫"]),
        tuple(["yousa", "你"]),
    ]

    for e, f in pair_tuples:
        t = Transition(m.tm.start, tuple([e, f]) , m.tm.start)
        if t not in m.tm.transitions_from[m.tm.start]:
            print("\tt(%s | %s) does not exist!!!!" % (f, e))
        print("t(%s | %s): %s" % (f, e, m.tm.transitions_from[m.tm.start][t]))
    return m


def partc(m: IBM1,
          mandarin_corpus: Sequence[Sequence[str]],
          english_corpus: Sequence[Sequence[str]],
          out_path: str,
          gold_path: str
          ) -> Mapping[str, float]:
    """TODO: This function should take a fully-trained IBM1 model and predict the best alignments
             for every pair of sequences in the mandarin_corpus and english_corpus. Your code
             should write these alignments in the following format:

                when processing a f_seq, e_seq pair, we want to produce a sequence of pairs.
                For every position j in f_seq and position i in e_seq, we want to produce
                a pair (j, i) if the token in f_seq at position j was aligned to the token in e_seq at position i.

                When using ibm1.predict(mandarin_corpus, english_corpus) you will receive
                an Iterable[Sequence[Tuple[int, int]]] data type. The outer Iterable will produce a sequence
                of alignment tuples. Please convert each tuple (j, i) to the format "{j}-{i}" and join them together
                to construct a space-separated string containing the alignments for a single sequence. Please
                write each alignment sequence to a separate line in out_path. For example, I would write the
                following alignment sequences:

                    [[(2,1), (1,3)], [(1, 1), (2, 0), (3, 2)]] to the out_path like so:

                    2-1 1-3
                    1-1 2-0 3-2

             Your code should then call the 'align_f1' function (you'll have to give it two paths and the ordering
             of the paths matters) to calculate your performance metrics on your alignments. Please return the
             mapping produced by 'align_f1'
    """

    print("part1.partc: TODO complete me!")
    return None

if __name__ == "__main__":
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")

    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    out_path = os.path.join(generated_dir, "part1_alignments.out")

    train_path = os.path.join(data_dir, "train.zh-en")
    gold_path = os.path.join(data_dir, "train.align")

    mandarin_corpus, english_corpus = parta(train_path)
    m = partb(mandarin_corpus, english_corpus)
    partc(m, mandarin_corpus, english_corpus, out_path, gold_path)

