from collections import defaultdict
from typing import Optional

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

    def __init__(self, sentence: str):
        self.parsed_sentence = parse(sentence)
        self.tree = self._build_tree()

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

    def __str__(self):
        def node_to_str(node, depth=0) -> str:
            return (
                f'{"  " * depth}{node.dep}[{node.lemma}]\n'
                + ''.join([node_to_str(child, depth+1) for child in node.children])
            )
        return node_to_str(self.tree)


def build_tree_from_map(root: DepTree.Node, tree_map: dict) -> DepTree.Node:
    if tree_map[root.id]:
        root.children = [
            build_tree_from_map(child, tree_map)
            for child in tree_map[root.id]
        ]
    return root


def deptree_similarity(one: DepTree, another: DepTree) -> float:
    pass
