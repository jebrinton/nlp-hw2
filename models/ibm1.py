# SYSTEM IMPORTS
from collections.abc import Iterable, Sequence, Mapping
from collections import Counter, defaultdict
from typing import Dict, Type, Tuple
import math
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from from_file import load_aligned_data
from fst import FST, Transition


# TYPES DECLARED IN THIS MODULE
IBM1Type: Type = Type["IBM1"]


# CONSTANTS
NULL_WORD: str = "NULL" # a special token to mean "no alignment"


class IBM1(object):
    def __init__(self: IBM1Type,
                 M: int = 100
                 ) -> None:
        self.tm: FST = FST()
        self.M: int = M

    def _add_null_word(self: IBM1Type,
                       data: Sequence[Sequence[str]]
                       ) -> Sequence[Sequence[str]]:

        # make a deepcopy of the input data where we've prepended the NULL_WORD to the beginning of every seq
        new_data: Sequence[Sequence[str]] = list()
        for seq in data:
            if isinstance(seq, str):
                new_data.append(NULL_WORD + " " + seq)
            else:
                new_data.append([NULL_WORD] + seq)
        return new_data

    def _init_tm(self: IBM1Type,
                 f_corpus: Sequence[Sequence[str]],
                 e_corpus: Sequence[Sequence[str]]
                 ) -> None:

        # Remember ibm1 has a single state
        start_state: str = "q0"
        self.tm.set_start(start_state)
        self.tm.set_accept(start_state)

        # for every aligned sequence, only create translation probabilities for
        # words that appear in the same sentence
        # set the weights to zero so we can reweight them later...just creating transition objects now
        # and adding them to the FST
        for f_seq, e_seq in zip(f_corpus, e_corpus):
            for f in f_seq:
                for e in e_seq:
                    self.tm.add_transition(Transition(self.tm.start,
                                                      tuple([e, f]),
                                                      self.tm.start), wt=0)
                self.tm.add_transition(Transition(self.tm.start,
                                                  tuple([NULL_WORD, f]),
                                                  self.tm.start), wt=0)
        # init uniform distribution
        for q in self.tm.states:
            for t in self.tm.transitions_from[q].keys():
                self.tm.reweight_transition(t, wt=1.0)

        self.tm.normalize_cond()

    def log_likelihood(self: IBM1Type,
                       f_seq: Sequence[str],
                       e_seq: Sequence[str]):
        ll = math.log(1.0) - math.log(self.M)
        for f in f_seq:
            f_total = 0.0
            for e in e_seq:
                t = Transition(self.tm.start,
                               tuple([e, f]),
                               self.tm.start)
                f_total += self.tm.transitions_from[self.tm.start][t]
            ll += math.log(f_total) - math.log(len(e_seq))
        return ll

    def estep(self: IBM1Type,
              f_corpus: Sequence[Sequence[str]],
              e_corpus: Sequence[Sequence[str]]
              ) -> Mapping[Tuple[str, str], float]:
        """TODO: implement the e-step for IBM1
                 You will want to calculate counts using the equations in the prompt.
                 Whenever you need to lookup a transition to get its weight,
                 you need to create a Transition object itself and then use it
                 to index into either the outgoing transition map or the incoming transition map
                 (IBM1 has only a single state).

                 To do this, lets say you construct a Transition object to lookup the 
                 weight of the transition ("原力", "force"):

                    t = Transition(self.tm.start,
                                   ("原力", "force"),
                                   self.tm.start)

                 you can either use this to index into the outgoing edges (self.tm.transitions_from)
                 or index into the incoming edges (self.tm.transition_to) to lookup the weight:

                    self.tm.transitions_from[self.tm.start][t]      # returns the weight of t

                 or:

                    self.tm.transitions_to[self.tm.accept][t]       # returns the weight of t

                 Your code should return a mapping key'd by the (input-vocab-element, output-vocab-element) pair
                 and should contain the value of the soft counts for each pair in the model.
        """

        counts: dict[Tuple[str, str], float] = defaultdict(float)
        
        for f_line, e_line in zip(f_corpus, e_corpus):
            for f_word in f_line:

                f_total = 0.0
                # compute sum over all e (denominator)
                for e_word in e_line:
                    # Transition takes (input_token, output_token)
                    t = Transition(self.tm.start, (e_word, f_word), self.tm.start)
                    f_total += self.tm.transitions_from[self.tm.start][t]

                if f_total == 0.0:
                    continue
                # add to counts
                for e_word in e_line:
                    t = Transition(self.tm.start, (e_word, f_word), self.tm.start)
                    counts[t.a] += self.tm.transitions_from[self.tm.start][t] / f_total

        return counts

    def mstep(self: IBM1Type,
              counts: Mapping[Tuple[str, str], float]
              ) -> None:
        """TODO: This method should accept the counts calculated during the E-step and use them to reweight
                 the transitions in self.tm. Be sure once you're done reweighting transitions, that you
                 call self.tm.normalize_cond() so that the FST code will normalize all of the pmfs for you!
        """
        # sub_sums = defaultdict(float)

        # for (f_word, e_word) in counts.keys():
        #     for (f_prime, _) in counts.keys():
        #         if (f_prime, e_word) in counts.keys():
        #             sub_sums[f_prime] += counts[(f_prime, e_word)]
        #     print("sub_sums created")

        #     t = Transition(self.tm.start, (f_word, e_word), self.tm.start)

        #     if sub_sums[f_word] != 0:
        #         if (f_word, e_word) in counts:
        #             self.tm.reweight_transition(t, wt = counts[(f_word, e_word)] / sub_sums[f_word])
        #             # self.tm.transitions_from[self.tm.start][t] = self.tm.transitions_to[self.tm.accept][t] = counts[(f_word, e_word)] / sub_sums[f_word]

        # if this doesn't work try reweighting transisitions

        # sub_sums = defaultdict(float)

        # for (e_word, f_prime) in counts.keys():
        #     if (e_word, f_prime) in counts.keys():
        #             sub_sums[f_prime] += counts[(e_word, f_prime)]

        # for q in self.tm.states:
        #     for t in self.tm.transitions_from[q].keys():
        #         (e_word, f_word) = t.a
        #         if (e_word, f_word) in counts.keys() and sub_sums[f_word] != 0:
        #             self.tm.reweight_transition(t, wt = counts[(e_word, f_word)] / sub_sums[f_word])

        for q in self.tm.states:
            for t in self.tm.transitions_from[q].keys():
                # (e_word, f_word) = t.a
                # if (e_word, f_word) in counts.keys():
                self.tm.reweight_transition(t, wt = counts[t.a])

        # also try—replacing (e_word, f_word) with t.a
        # or printing t.a to verify it is the same as the tuple you created

        # this will do everything you did in subsums
        self.tm.normalize_cond()
        # print("m", end="", flush=True)

    def _train(self: IBM1Type,
               f_corpus: Sequence[Sequence[str]],
               e_corpus: Sequence[Sequence[str]],
               max_iter: int = int(1e6),
               converge_threshold: float = 1e-5,
               ) -> Sequence[float]:

        # collect log-likelihood as a function of iteration. These should monotonically increase (we will check)
        lls: Sequence[float] = list()

        # when we train EM its often nice to measure error as "percent relative error"
        # which measures the difference between two iterations as a percent of what the current iteration's value is
        # This gets around the "scale" issue of deciding a threshold. If we say "threshold = 1e-4", well if the
        # values we're using to determine convergence are in a similar scale, then a threshold of 1e-4 might be kind
        # of big! By measuring difference as a percentage of the current iteration, we remove this scaling problem.
        current_iter = 0
        prev_error = 0.0
        current_rel_error = 1

        # if you set converge_threshold to None you can force EM to run for 'max_iter' # of iterations
        # idk why you'd want to do this but nevertheless its an option
        if converge_threshold is None:
            converge_threshold = math.inf

        while current_iter < max_iter and current_rel_error > converge_threshold:
            # E step
            counts = self.estep(f_corpus, e_corpus)

            # M step
            self.mstep(counts)

            # measure log-likelihood after every EM update
            ll = 0.0
            for f_seq, e_seq in zip(f_corpus, e_corpus):
                ll += self.log_likelihood(f_seq, e_seq)

            # measure the new relative error and shift variables
            current_rel_error = abs(ll - prev_error) / abs(ll)
            prev_error = ll
            current_iter += 1
            lls.append(ll)

        return lls

    def train_from_file(self: IBM1Type,
                        aligned_data_path: str,
                        aligned_data_delimiter: str = "\t",
                        max_iter: int = int(1e6),
                        converge_threshold: float = 1e-5,
                        ) -> Sequence[str]:
        # load the data
        f_corpus, e_corpus = load_aligned_data(aligned_data_path, delimiter=aligned_data_delimiter)

        return self.train_from_raw(f_corpus,
                                   e_corpus,
                                   max_iter=max_iter,
                                   converge_threshold=converge_threshold)

    def train_from_raw(self: IBM1Type,
                       f_corpus: Sequence[Sequence[str]],
                       e_corpus: Sequence[Sequence[str]],
                       max_iter: int = int(1e6),
                       converge_threshold: float = 1e-5,
                       ) -> Sequence[float]:
        # add the null word to the e corpus
        e_corpus = self._add_null_word(e_corpus)

        # initialize the model
        self._init_tm(f_corpus, e_corpus)

        # run em
        return self._train(f_corpus,
                           e_corpus,
                           max_iter=max_iter,
                           converge_threshold=converge_threshold)

    def predict(self: IBM1Type,
                f_corpus: Sequence[Sequence[str]],
                e_corpus: Sequence[Sequence[str]]
                ) -> Iterable[Sequence[Tuple[int, int]]]:
        # add the null word to the e corpus
        e_corpus = self._add_null_word(e_corpus)

        # yield the prediction for every pair of sequences in the aligned corpus
        for f_seq, e_seq in zip(f_corpus, e_corpus):
            yield self.predict_seq(f_seq, e_seq)

    def predict_seq(self: IBM1Type,
                    f_seq: Sequence[str],
                    e_seq: Sequence[str]
                    ) -> Sequence[Tuple[int, int]]:
        # calculate alignments
        alignment_pairs = list()
        for j, f in enumerate(f_seq):

            # find the best alignment for f token "f" at position "j"
            best_alignment = 0
            best_translation_prob = 0.0

            # consider each e token "e" at position "i"
            for i, e in enumerate(e_seq):

                # make transition object so we can look it up in the fst
                t = Transition(self.tm.start, tuple([e, f]), self.tm.start)

                # argmax
                if self.tm.transitions_from[self.tm.start][t] > best_translation_prob:
                    best_alignment = i
                    best_translation_prob = self.tm.transitions_from[self.tm.start][t]

            # don't need to produce an alignment for the NULL_WORD so adjust alignment to be 0-indexed again
            # (with the NULL_WORD present the alignment is 1-indexed)
            if e_seq[best_alignment] != NULL_WORD:
                alignment_pairs.append(tuple([j, best_alignment-1]))

        return alignment_pairs

