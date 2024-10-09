# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from collections import Counter, defaultdict
from typing import Tuple, Type, Union
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from utils import Vocab, START_TOKEN, END_TOKEN, UNK_TOKEN
from fst import FST, Transition, EPSILON


# TYPES DECLARED IN THIS MODULE
UniformType: Type = Type["Uniform"]
KneserNeyType: Type = Type["KneserNey"]


class Uniform(object):
    """Uniform distribution."""
    def __init__(self: UniformType,
                 data: Sequence[Sequence[str]]
                 ) -> None:
        self.vocab = Vocab()
        for seq in data:
            vocab |= seq

    def prob(self: UniformType,
             q: Sequence[str],
             w: str):
        return 1/(len(self.vocab) - 1) # ignoring START_TOKEN


class KneserNey(object):
    def __init__(self: KneserNeyType,
                 data: Sequence[Sequence[str]],
                 n: int,
                 bom: Union[KneserNeyType, UniformType] = None):
        self.bom = bom
        
        # Collect n-gram counts
        cqw = defaultdict(Counter)
        cq = Counter()
        for line in data:
            q = (START_TOKEN,)*(n-1)
            for w in line + [END_TOKEN]:
                cqw[q][w] += 1
                cq[q] += 1
                q = (q+(w,))[1:]

        # Compute discount
        cc = Counter()
        for q in cqw:
            for w in cqw[q]:
                cc[cqw[q][w]] += 1
        d = cc[1] / (cc[1] + 2*cc[2])

        # Compute probabilities and backoff weights
        self._prob = defaultdict(dict)
        self._bow = {}
        for q in cqw:
            for w in cqw[q]:
                self._prob[q][w] = (cqw[q][w]-d) / cq[q]
            self._bow[q] = len(cqw[q])*d / cq[q]

    def prob(self: KneserNeyType,
             q: Sequence[str],
             w: str):
        if q in self._prob:
            return self._prob[q].get(w, 0) + self._bow[q] * self.bom.prob(q[1:], w)
        else:
            return self.bom.prob(q[1:], w)

def make_kneserney(data: Sequence[Sequence[str]],
                   n: int
                   ) -> FST:
    """Create a Kneser-Ney smoothed language model of order `n`, 
    trained on `data`, as a `FST`.
    Note that the returned FST has epsilon transitions. To iterate
    over states in topological order, sort them using `lambda q:
    -len(q)` as the key.
    """

    # Estimate KN-smoothed models for orders 1, ..., n
    kn = {}
    for i in range(1, n+1):
        kn[i] = KneserNey(data, i)

    # Create the FST. It has a state for every possible k-gram for k = 0, ..., n-1.
    m = FST()
    m.set_start((START_TOKEN,) * (n-1))
    m.set_accept((END_TOKEN,))
    
    for i in range(1, n+1):
        for u in kn[i]._prob:
            if i > 1:
                # Add an epsilon transition that backs off from the i-gram model to the (i-1)-gram model
                m.add_transition(Transition(u, (EPSILON, EPSILON), u[1:]), kn[i]._bow[u])
            else:
                # Smooth 1-gram model with uniform distribution
                types = len(kn[i]._prob[u])+1
                for w in kn[i]._prob[u]:
                    m.add_transition(Transition(u, (w, w), (w,)), 1/types)
                m.add_transition(Transition(u, (UNK_TOKEN, UNK_TOKEN), ()), 1/types)

            # Create transitions for word probabilities
            for w in kn[i]._prob[u]:
                # If we are in state u and read w, then v is the new state.
                # This should be the longest suffix of uw that is observed
                # in the training data.
                if w == END_TOKEN:
                    v = (END_TOKEN,)
                else:
                    v = u+(w,)
                    while len(v) > 0 and (len(v) >= n or v not in kn[len(v)+1]._prob):
                        v = v[1:]
                m.add_transition(Transition(u, (w, w), v), kn[i]._prob[u][w])
    return m

if __name__ == "__main__":
    # This demonstrates how to create a Kneser-Ney smoothed language
    # model by directly using the KneserNey class, without using FSTs.

    n = 3

    data = []
    for line in fileinput.input():
        words = line.split()
        data.append(words)

    lm = Uniform(data)
    for i in range(1, n+1):
        lm = KneserNey(data, i, lm)

