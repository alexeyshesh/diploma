from collections import defaultdict
from typing import List

import spacy_udpipe

spacy_udpipe.download('ru')
parse = spacy_udpipe.load('ru')


class DepTree:
    class Node:
        def __init__(self, token: dict):
            self.id = token['id']
            self.lemma = token['lemma']
            self.pos = token['pos']
            self.dep = token['dep']
            self.children = []
            self.height = 0

        def __eq__(self, other):
            res = (
                self.id == other.id
                and self.lemma == other.lemma
                and self.pos == other.pos
                and self.dep == other.dep
                and len(self.children) == len(other.children)
            )

            for i in range(len(self.children)):
                res = res and (self.children[i] == other.children[i])

            return res

        def __str__(self):
            return f'{self.dep}' + (
                '' if len(self.children) == 0
                else f'({" ".join([f"{str(child)}" for child in self.children])})'
            )

    def __init__(self, sentence: str):
        self.parsed_sentence = parse(sentence)
        self.tree = self._build_tree()

    def __str__(self):
        return str(self.tree)

    def _build_tree(self) -> Node:
        tokens = self.parsed_sentence.to_json()['tokens']
        tree_map = defaultdict(list)
        root = None
        for token in tokens:
            if token['id'] == token['head']:
                root = self.Node(token)
                continue
            tree_map[token['head']].append(self.Node(token))

        return build_tree_from_map(root, tree_map)


def build_tree_from_map(root: DepTree.Node, tree_map: dict) -> DepTree.Node:
    if tree_map[root.id]:
        root.children = [
            build_tree_from_map(child, tree_map)
            for child in tree_map[root.id]
        ]
        root.height += max([x.height for x in root.children])
    return root


def deptree_similarity(one: DepTree, other: DepTree) -> float:
    return node_similarity(one.tree, other.tree)


def node_similarity(one: DepTree.Node, other: DepTree.Node) -> float:
    if one.dep == other.dep:
        return (1 + _node_children_similarity(one.children, other.children)) / 2
    else:
        return 0


def _node_children_similarity(one: List[DepTree.Node], other: List[DepTree.Node]) -> float:
    if len(one) * len(other) == 0:
        if len(one) == len(other):
            return 1
        return 0

    if len(one) == len(other):
        size = len(one)
        return sum([node_similarity(one[i], other[i]) for i in range(size)]) / size
    else:
        if len(one) > len(other):
            one, other = other, one
        size = len(one)
        return max([
            _node_children_similarity(one, other[i:size+i])
            for i in range(size)
        ])
