# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from typing import Tuple, Type


# PYTHON PROJECT IMPORTS
from fst import FST, Transition, StateType


# TYPES DECLARED IN THIS MODULE
SearchColorsType: Type = Type["SearchColors"]
TopologicalPacketType: Type = Type["TopologicalPacket"]


class SearchColors(object):
    WHITE = 0
    GRAY = 1
    BLACK = 2


class TopologicalPacket(object):
    def __init__(self):
        self.color: SearchColors = SearchColors.WHITE
        self.discovery_time: int = 0
        self.finishing_time: int = 0
        self.parent: TopologicalPacket = None


def fst_dfs_visit(fst: FST,
                  state_dfs_map: Mapping[StateType, TopologicalPacket],
                  order: Sequence[StateType],
                  state: StateType,
                  time: int
                  ) -> None:
    time += 1

    # discover the node
    state_dfs_map[state].discovery_time = time
    state_dfs_map[state].color = SearchColors.GRAY

    def sorter(x: Tuple[Transition, float]):
        # print(x)
        t, wt = x
        return t.q[0][0], -len(t.q[1])

    sorted_order = sorted(fst.transitions_from[state].items(),
                          key=lambda x: sorter(x))
    for transition, _ in sorted_order:
        # visit all unvisited children
        if state_dfs_map[transition.r].color == SearchColors.WHITE:
            state_dfs_map[transition.r].parent = state
            fst_dfs_visit(fst, state_dfs_map, order, transition.r, time)
    # finish the node
    state_dfs_map[state].color = SearchColors.BLACK
    time += 1
    state_dfs_map[state].finishing_time = time
    order.insert(0, state)


def fst_dfs_search(fst: FST,
                   order: Sequence[StateType]
                   ) -> None:
    time = 0

    # make sure we unvisit all nodes
    state_dfs_map = {name: TopologicalPacket() for name in fst.states}

    # perform the depth first search
    # for state in fst.states:
    #     if state_dfs_map[state].color == SearchColors.WHITE:
    #         fst_dfs_visit(fst, state_dfs_map, order, state, time)
    fst_dfs_visit(fst, state_dfs_map, order, fst.start, time)


def fst_topsort(fst: FST) -> Sequence[StateType]:
    topological_order = list()
    fst_dfs_search(fst, topological_order)
    return topological_order    

