# SYSTEM IMPORTS
from collections.abc import Iterable, Mapping, Sequence
from typing import Type, Tuple
import collections
import numpy as np
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from from_file import load_aligned_data
from fst import FST, Transition, StateType, compose, create_seq_fst, EPSILON
from topsort import fst_topsort
from utils import START_TOKEN, END_TOKEN
from ibm1 import IBM1, NULL_WORD
from lm import make_kneserney



# TYPES DECLARD IN THIS MODULE
TranslatorType: Type = Type["Translator"]



class Translator(object):
    def __init__(self: TranslatorType,
                 M: int = 100
                 ) -> None:
        self.lm: FST = FST()
        self.tm: FST = FST()
        self.M: int = M

    def _train_lm(self: TranslatorType,
                  e_corpus: Sequence[Sequence[str]],
                  n: int = 2
                  ) -> None:
        self.lm = make_kneserney(e_corpus, n)

    def use_pretrained_tm(self: TranslatorType,
                          pretrained_ibm1: IBM1,
                          num_rules_to_keep: int = 10,
                          default_unk_prob: float = 10e-100,
                          unknown_fs: Sequence[str] = None
                          ) -> None:

        if unknown_fs is None:
            unknown_fs = set()

        # initialize another ibm1 fst but only keeping "num_rules_to_keep" transitions per output token
        # we are also going to flip the input/output vocabs for this secondary ibm1 model
        # so while the original ibm1 has transitions for (e, f) tokens (i.e. read e and emit f)
        # we are going to flip these and store transitions for (f,e) pairs instead.
        # The reason for this is when we do our composition with the three FSTS:
        #   - ibm1
        #   - lm
        #   - fst for a sequence w (containing f tokens)
        # we can try to get a smaller output FST.
        #
        # For instance, if we did the composition order compose(compose(lm, ibm1), fst_w)
        # we would be generating a massive FST for the output of compose(lm, imb1) which will be reduced
        # to a smaller FST when composed with fst_w
        #
        # instead we can do this: compose(compose(fst_w, reversed_transitions(ibm1)), lm)
        # which will take in w and emit all possible sequences that could generate w
        # This FST will be smaller than the other one, meaning that it will run faster
        start_state = "q0"
        end_state = "q1"
        self.tm.set_start(start_state)
        self.tm.set_accept(end_state)
        self.tm.add_transition(Transition(start_state,
                                          tuple([END_TOKEN, END_TOKEN]),
                                          end_state),
                                wt=1)  # wt=1 is implicit but I'll include it here for clarity

        # collect rules together from the ibm1 model that have the same output token
        f_transition_dict = collections.defaultdict(list)
        for q in pretrained_ibm1.tm.states:
            for t, wt in pretrained_ibm1.tm.transitions_from[q].items():
                _, f = t.a
                f_transition_dict[f].append(tuple([t, wt]))

        # sort each list by prob and add the top "num_rules_to_keep" to the secondary ibm1 fst
        for t_wt_list in f_transition_dict.values():
            sorted_list = sorted(t_wt_list, key=lambda x: x[1], reverse=True)[:num_rules_to_keep]
            for t, wt in sorted_list:
                e, f = t.a

                # convert NULL rules into deletion rules (i.e. this token in f_seq does not align to an e token)
                if e == NULL_WORD:
                    e = EPSILON

                self.tm.add_transition(Transition(self.tm.start,
                                                  tuple([f, e]), # flip transition when we copy it over
                                                  self.tm.start),
                                       wt=wt)


        # if we have unknown_f tokens, give them transitions where we delete them
        for f in unknown_fs:
            self.tm.add_transition(Transition(self.tm.start,
                                              tuple([f, EPSILON]),
                                              self.tm.start),
                                   wt=default_unk_prob)

    def _train_tm(self: TranslatorType,
                  f_corpus: Sequence[Sequence[str]],
                  e_corpus: Sequence[Sequence[str]],
                  max_iter: int = 20,
                  num_rules_to_keep: int = 10,
                  default_unk_prob: float = 10e-100,
                  converge_threshold: float = 1e-5,
                  unknown_fs: Sequence[str] = None,
                  ) -> None:

        ibm1_tm = IBM1(M=self.M)
        ibm1_tm.train_from_raw(f_corpus,
                               e_corpus,
                               max_iter=max_iter,
                               converge_threshold=converge_threshold)

        self.use_pretrained_tm(ibm1_tm,
                               num_rules_to_keep=num_rules_to_keep,
                               default_unk_prob=default_unk_prob,
                               unknown_fs=unknown_fs)

        

    def _train(self: TranslatorType,
               f_corpus: Sequence[Sequence[str]],
               e_corpus: Sequence[Sequence[str]],
               lm_n: int = 2,
               tm_max_iter: int = int(1e6),
               num_rules_to_keep: int = 10,
               default_unk_prob: float = 10e-100,
               converge_threshold=1e-5,
               unknown_fs: Sequence[str] = None
               ) -> None:

        # train lm
        self._train_lm(e_corpus,
                       n=lm_n)

        # train ibm1 and extract rules from it to populate our translation model
        self._train_tm(f_corpus,
                       e_corpus,
                       max_iter=tm_max_iter,
                       num_rules_to_keep=num_rules_to_keep,
                       default_unk_prob=default_unk_prob,
                       converge_threshold=converge_threshold,
                       unknown_fs=unknown_fs)

    def train_from_file(self: TranslatorType,
                        aligned_fpath: str,
                        lm_n: int = 2,
                        tm_max_iter: int = int(1e6),
                        num_rules_to_keep: int = 10,
                        default_unk_prob: float = 10e-100,
                        converge_threshold: float = 1e-5,
                        unknown_fs: Sequence[str] = None
                        ) -> TranslatorType:

        # load the corpus
        f_corpus, e_corpus = load_aligned_data(aligned_fpath)

        # train the model
        return self.train_from_raw(f_corpus, e_corpus,
                                   tm_max_iter=tm_max_iter,
                                   num_rules_to_keep=num_rules_to_keep,
                                   default_unk_prob=default_unk_prob,
                                   converge_threshold=converge_threshold,
                                   unknown_fs=unknown_fs)
        return self

    def train_from_raw(self: TranslatorType,
                       f_corpus: Sequence[Sequence[str]],
                       e_corpus: Sequence[Sequence[str]],
                       lm_n: int = 2,
                       tm_max_iter: int = 100,
                       num_rules_to_keep: int = 10,
                       default_unk_prob: int = 10e-100,
                       converge_threshold: float = 1e-5,
                       unknown_fs: Sequence[str] = None
                       ) -> TranslatorType:
        self._train(f_corpus,
                    e_corpus,
                    tm_max_iter=tm_max_iter,
                    num_rules_to_keep=num_rules_to_keep,
                    default_unk_prob=default_unk_prob,
                    converge_threshold=converge_threshold,
                    unknown_fs=unknown_fs)
        return self

    def viterbi(self: TranslatorType,
                fst: FST
                ) -> Tuple[Sequence[StateType], float, Sequence[Transition]]:
        """TODO: The viterbi algorithm
                 Your code should perform find the maximum-weighted path through this graph
                 using the topological ordering of the vertices (which I have calculated for you below).
                 If you want to look at incoming edges, you should use the fst.transitions_to field which
                 provides a two-level map. The first key is the state, and the value is a mapping between
                 transition objects and their weight in the FST.

                 For example, you could use fst.transitions_to[fst.accept] to lookup all transitions that end
                 at the accept state of this FST (in NLP there is only a single accept state but in general
                 there may be more).

                 I would recommend that instead of doing the normal viterbi comparison where we multiply
                 probabilities, that you add logprobs instead. So for instance, instead of doing:

                    viterbi[q] * p > viterbi[q']

                 you would do:

                    viterbi[q] + np.log(p) > viterbi[q']

                 Remember that some of these probs will be exceedingly small, so multiplication may not be the
                 best strategy here.

                 Your code should return a triple:
                    - the path of states (from fst.start -> fst.accept) with the max logprob
                    - the logprob of the max path
                    - the transitions that were used in the max path

                One note about the actual path (the first element of your return-triple) is that it should not
                only include states that do not have the START_TOKEN in the state-name itself nor should you
                include fst.start itself. We only want the states that have the generated tokens in them,
                so when you are walking your backpointers, or whatever mechanism you use to reconstruct the path
                make sure to stop the reconstruction when you encounter START_TOKEN in state[1][-1] (the state will be
                a tuple with the following structure ((x, y), z) where:
                    - x is a state from the fst for input sequence w
                    - y is a state from ibm1 (i.e. "q0" or "q1")
                    - z is a state from the language model (the states of which are the gram....also a tuple of str)

                this is a result of FST composition. So, remember to stop your reconstruction when either of these
                conditions are met:
                    - the state you're thinking of adding is fst.start
                    - START_TOKEN appears in state[1][-1]
                        - be warned! state[1] could be an empty tuple (a result of the smoothing in KneserNey)
        """
        state_topological_order: Sequence[StateType] = fst_topsort(fst)

        return None, None, None

    def compose(self: TranslatorType,
                *list_of_fsts: Sequence[FST]
                ) -> FST:

        if len(list_of_fsts) < 2:
            raise ValueError("cannot compose less than 2 fsts")

        composed_fst = list_of_fsts[0]
        for fst in list_of_fsts[1:]:
            composed_fst = compose(composed_fst, fst)

        return composed_fst

    def decode_seq(self: TranslatorType,
                   f_seq: str
                   ) -> Tuple[str, float]:
        seq_fst: FST = create_seq_fst(f_seq)

        # remember we reversed the order of composition so that we would get a smaller FST
        # which means our algorithm will run faster
        fst = self.compose(seq_fst, self.tm, self.lm)
        state_path, log_prob, _ = self.viterbi(fst)

        return ' '.join([s[-1][-1] for s in state_path[:-1] if len(s[-1]) > 0]), log_prob

    def decode(self: TranslatorType,
               f_corpus: Sequence[Sequence[str]],
               num_seq_to_decode: int = None
               ) -> Iterable[Tuple[str, float]]:
        if num_seq_to_decode == None:
            num_seq_to_decode = len(f_corpus)

        for i, f_seq in enumerate(f_corpus):
            if i >= num_seq_to_decode:
                return
            yield self.decode_seq(f_seq)

