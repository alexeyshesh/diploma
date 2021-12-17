from abc import ABC, abstractmethod
from typing import List

from src.deptree import DepTree, deptree_similarity


class AbstractCluster(ABC):
    @abstractmethod
    def clusterize(self, trees: List[DepTree]) -> List[set]:
        pass


class BasicCluster(AbstractCluster):
    def __init__(self, threshold):
        self.threshold = threshold

    def clusterize(self, trees: List[DepTree]) -> List[set]:
        total_len = len(trees)
        trees = (set(trees))
        clusters = []
        while len(trees) > 0:
            print(f'ready {(len(trees) / total_len * 100):.2}%')
            core = set()
            new_members = {next(iter(trees))}
            while new_members:
                tmp = set()
                for tree in new_members:
                    for other_tree in trees:
                        if deptree_similarity(tree, other_tree) > self.threshold:
                            tmp.add(other_tree)
                core |= new_members
                trees -= tmp
                new_members = tmp
            clusters.append(core)

        return clusters
