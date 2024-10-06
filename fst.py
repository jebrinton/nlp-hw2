# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from collections import Counter, defaultdict
from typing import Tuple, Type, Union
import itertools
import math


# PYTHON PROJECT IMPORTS
from utils import END_TOKEN


# CONSTANTS
EPSILON="ε"                                 # used to refer to insertion/deletion transitions


# TYPES DECLARED IN THIS MODULE
StateType: Type = Union[str, Tuple]         # states can either be strings or, when we compose them together, tuples
TransitionType: Type = Type["Transition"]   # the type for the transition class
FSTType: Type = Type["FST"]                 # the type for the fst class


class Transition(object):
    def __init__(self: TransitionType,
                 q: StateType,              # could be a str or a tuple (the elements of which are str or tuple)
                 a: Tuple[str, str],        # the tokens of the transition (input_token, output_token)
                 r: StateType               # could be a str or a tuple (the elements of which are str or tuple)
                 ) -> None:
        self.q: StateType = q
        self.a: Tuple[str, str] = a
        self.r: StateType = r

    def __eq__(self: TransitionType,
               other: object
               ) -> bool:
        return (isinstance(other, Transition) and
                (self.q, self.a, self.r) == (other.q, other.a, other.r))

    def __ne__(self: TransitionType,
               other: object
               ) -> bool:
        return not self == other

    def __hash__(self: TransitionType) -> int:
        return hash((self.q, self.a, self.r))


class FST(object):
    def __init__(self: FSTType) -> None:

        # the states in this fst
        self.states: Sequence[str] = set()

        # outgoing transitions from states
        # given a state, you can lookup all outgoing edges from that state with self.transitions_from[state]
        # which returns a dictionary key'd by the transition object and value'd by the weight of that transition
        self.transitions_from: Mapping[StateType, Mapping[TransitionType, float]] = defaultdict(lambda: defaultdict(float))

        # incoming transitions to states
        # given a state, you can lookup all incoming edges from that state with self.transitions_to[state]
        # which returns a dictionary key'd by the transition object and value'd by the weight of that transition
        self.transitions_to: Mapping[StateType, Mapping[TransitionType, float]] = defaultdict(lambda: defaultdict(float))

        # transitions using a specific token in the input alphabet
        # given an input token, you can lookup all transitions using that token with self.transitions_on[token]
        # which returns a dictionary key'd by the transition object and value'd by the weight of that transition
        self.transitions_on: Mapping[str, Mapping[TransitionType, float]] = defaultdict(lambda: defaultdict(float))

        # start state
        self.start: StateType = None

        # accept state (there can be multiple in a general fst but not in our use case)
        self.accept: StateType = None

        # input vocab
        self.input_alphabet: set = set()

        # output vocab
        self.output_alphabet: set = set()

    def add_state(self: FSTType,
                  q: StateType
                  ) -> None:
        """Adds state q."""
        self.states.add(q)

    def set_start(self: FSTType,
                  q: StateType
                  ) -> None:
        """Sets the start state to q."""
        self.add_state(q)
        self.start = q

    def set_accept(self: FSTType,
                   q: StateType
                   ) -> None:
        """Sets the accept state to q."""
        self.add_state(q)
        self.accept = q

    def add_transition(self: FSTType,
                       t: TransitionType,
                       wt: float = 1
                       ) -> None:
        """Adds the transition 
             a:b/wt
           q ------> r
        
        If q and r are not already states, they are added too.
        If t is already a transition, its weight is incremented by wt."""
        self.add_state(t.q)
        self.add_state(t.r)
        self.input_alphabet.add(t.a[0])
        self.output_alphabet.add(t.a[1])
        self.transitions_from[t.q][t] += wt
        self.transitions_to[t.r][t] += wt
        self.transitions_on[t.a[0]][t] += wt

    def reweight_transition(self: FSTType,
                            t: TransitionType,
                            wt: float = 1
                            ) -> None:
        """Replaces the weight of transition t with new weight wt."""
        # To do: eliminate this in favor of a separate self.weight[t]
        self.transitions_from[t.q][t] = wt
        self.transitions_to[t.r][t] = wt
        self.transitions_on[t.a[0]][t] = wt

    def train_joint(self: FSTType,
                    data: Sequence[Sequence[str]]
                    ) -> None:
        """Trains the transducer on the given data."""
        c = Counter()
        alphabet = set()
        for line in data:
            q = self.start
            for a in list(line) + [END_TOKEN]:
                for t in self.transitions_from[q]:
                    if a == t.a[0]:
                        c[t] += 1
                        q = t.r
                        break
                else:
                    raise ValueError("training string is not in language")
        for q in self.states:
            z = sum(self.transitions_from[q].values())
            for t in self.transitions_from[q]:
                self.reweight_transition(t, c[t]/z)

    def normalize_joint(self: FSTType) -> None:
        """Renormalizes weights so that path weights form a joint
        probability distribution (input and output)."""
        for q in self.states:
            s = Counter()
            z = 0
            for t, wt in self.transitions_from[q].items():
                s[t.a[0]] += wt
                z += wt
            for t, wt in self.transitions_from[q].items():
                if wt == 0: continue
                self.reweight_transition(t, wt/z)

    def normalize_cond(self: FSTType,
                       add: float = 0
                       ) -> None:
        """Renormalizes weights so that path weights form a conditional
        probability distribution (output given input)."""
        for q in self.states:
            s = Counter()
            z = 0
            for t, wt in self.transitions_from[q].items():
                s[t.a[0]] += wt+add
                z += wt+add
            for t, wt in self.transitions_from[q].items():
                if wt+add == 0: continue
                if t.a[0] == EPSILON:
                    self.reweight_transition(t, (wt+add)/z)
                else:
                    self.reweight_transition(t, (wt+add)/s[t.a[0]]*(1-s[EPSILON]/z))

    def visualize(self: FSTType) -> None:
        """Pops up a window showing a transition diagram.
        Requires graphviz.
        Under MacOS Sierra, you'll need to upgrade to 10.12.2 or newer."""
        
        import subprocess
        from tkinter import (Tk, Canvas, PhotoImage,
                             Scrollbar, HORIZONTAL, VERTICAL, X, Y,
                             BOTTOM, RIGHT, LEFT, YES, BOTH, ALL)
        import base64

        def escape(s):
            return '"{}"'.format(s.replace('\\', '\\\\').replace('"', '\\"'))
        
        dot = []
        dot.append("digraph {")
        dot.append('START[label="",shape=none];')
        index = {}
        for i, q in enumerate(self.states):
            index[q] = i
            attrs = {'label': escape(str(q)), 'fontname': 'Courier'}
            if q == self.accept:
                attrs['peripheries'] = 2
            dot.append('{} [{}];'.format(i, ",".join("{}={}".format(k, v) for (k, v) in attrs.items())))
        dot.append("START->{};".format(index[self.start]))
        for q in self.states:
            ts = collections.defaultdict(list)
            for t in self.transitions_from.get(q, []):
                ts[t.r].append(t)
            for r in ts:
                if len(ts[r]) > 8:
                    label = "\n".join(":".join(map(str, t.a)) for t in ts[r][:5]) + "\n..."
                else:
                    label = "\n".join(":".join(map(str, t.a)) for t in ts[r])
                dot.append('{}->{}[label={},fontname=Courier];'.format(index[q], index[r], escape(label)))
        dot.append("}")
        dot = "\n".join(dot)
        proc = subprocess.run(["dot", "-T", "gif"], input=dot.encode("utf8"), stdout=subprocess.PIPE)
        if proc.returncode == 0:
            root = Tk()
            scrollx = Scrollbar(root, orient=HORIZONTAL)
            scrollx.pack(side=BOTTOM, fill=X)
            scrolly = Scrollbar(root, orient=VERTICAL)
            scrolly.pack(side=RIGHT, fill=Y)

            canvas = Canvas(root)
            image = PhotoImage(data=base64.b64encode(proc.stdout))
            canvas.create_image(0, 0, image=image, anchor="nw")
            canvas.pack(side=LEFT, expand=YES, fill=BOTH)
            
            canvas.config(xscrollcommand=scrollx.set, yscrollcommand=scrolly.set)
            canvas.config(scrollregion=canvas.bbox(ALL))
            scrollx.config(command=canvas.xview)
            scrolly.config(command=canvas.yview)

            root.mainloop()

def compose(m1: FSTType,
            m2: FSTType
            ) -> FSTType:
    """Compose two finite transducers m1 and m2, feeding the output of m1
    into the input of m2.

    In the resulting transducer, each transition t contains extra
    information about where it came from in the attribute
    t.composed_from:

    - (t1, t2) means that t simulates m1 following transition t1 and
      m2 following transition t2.
    - (t1, None) means that t simulates m1 following transition t1 and
      m2 doing nothing.
    - (None, t2) means that t simulates m1 doing nothing and m2
      following transition t2.
    """
    
    m = FST()
    m1_deletes = False
    m2_inserts = False

    m.set_start((m1.start, m2.start))
    for a in m1.transitions_on:
        for t1, wt1 in m1.transitions_on[a].items():
            if t1.a[1] != EPSILON:
                for t2, wt2 in m2.transitions_on.get(t1.a[1], {}).items():
                    t = Transition((t1.q, t2.q), (t1.a[0], t2.a[1]), (t1.r, t2.r))
                    t.composed_from = (t1, t2)
                    m.add_transition(t, wt=wt1*wt2)
            else:
                m1_deletes = True
                for q2 in m2.states:
                    t = Transition((t1.q, q2), (t1.a[0], EPSILON), (t1.r, q2))
                    t.composed_from = (t1, None)
                    m.add_transition(t, wt=wt1)
    for q1 in m1.states:
        for t2, wt2 in m2.transitions_on.get(EPSILON, {}).items():
            m2_inserts = True
            t = Transition((q1, t2.q), (EPSILON, t2.a[1]), (q1, t2.r))
            t.composed_from = (None, t2)
            m.add_transition(t, wt=wt2)
    m.set_accept((m1.accept, m2.accept))
    """
    # If m1 is a deleting FST and m2 is an inserting FST,
    # the resulting composed FST has duplicate paths.
    # But if we're just search for the best path, this doesn't matter,
    # so this is commented out.
    if m1_deletes and m2_inserts:
        raise ValueError("Can't compose a deleting FST with an inserting FST")"""
    return m


def create_seq_fst(w: str) -> FST:
    """Creates a fst from a string. The created fst simply copies its input to its output.
       Use this function to create a fst for a specific string and then compose it with
       your model fst
    """
    data = list(w) + [END_TOKEN]

    model = FST()
    model.set_start(0)

    for i, char in enumerate(data):
        model.add_transition(Transition(i, (char, char), i+1))

    model.set_accept(len(data))
    return model

