from collections import defaultdict
from typing import List
from tqdm import tqdm

from nltk import sent_tokenize
from scipy.spatial.distance import cosine
import spacy_udpipe

spacy_udpipe.download('ru')
parse = spacy_udpipe.load('ru')


class DepTree:
    class Error(Exception):
        pass

    class Node:
        def __init__(self, token: dict):
            self.id = token['id']
            self.lemma = token['lemma']
            self.pos = token['pos']
            self.dep = token['dep']
            self.children = []

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
        if not sentence:
            raise self.Error('Sentence is empty')

        self.parsed_sentence = parse(sentence)
        self.tree = self._build_tree()
        self.tree_str = str(self.tree)

    def __str__(self):
        return str(self.tree)

    def __hash__(self):
        return hash(self.tree_str)

    def __eq__(self, other):
        return self.tree_str == other.tree_str

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


def build_deptree_vector(text: str, normalize=True) -> dict:
    sents = sent_tokenize(text)
    result = defaultdict(int)
    total_sents = len(sents)
    for i in tqdm(range(total_sents)):
        tree = DepTree(sents[i])
        result[tree] += 1

    if normalize:
        for sent in result:
            result[sent] /= total_sents

    return dict(result)


def text_vectors_similarity(first: dict, second: dict, sim_f=cosine):
    v1 = []
    v2 = []

    unique_second_vector_keys = set(second.keys())

    for elem in first:
        v1.append(first[elem])
        v2.append(second.get(elem, 0))
        unique_second_vector_keys.remove(elem)

    for elem in unique_second_vector_keys:
        v1.append(0)
        v2.append(second[elem])

    return sim_f(v1, v2)


def collect_children(structure: dict, root):
    result = set()
    cur_set = set(structure[root])
    while cur_set:
        tmp = set()
        for elem in cur_set:
            tmp |= set(structure.get(elem, []))
        result |= cur_set
        cur_set = tmp
    return result


def build_high_level_structure(sentence):
    tokens = parse(sentence).to_json()['tokens']
    return _build_high_level_structure(tokens)['short']


def get_sentence_hash(sentence):
    tokens = parse(sentence).to_json()['tokens']
    return _build_high_level_structure(tokens)['hash']


def _build_high_level_structure(tokens):
    # поиск деепричастных оборотов
    structure = defaultdict(list)
    for token in tokens:
        structure[token['head']].append(token['id'])

    high_level_structure = []
    cur_part = {
        'type': None,
        'text': [],
    }
    used = set()
    for token in tokens:
        if token['id'] in used:
            continue

        if token['dep'] == 'advcl':
            if cur_part['type'] != 'participial_turnover' and cur_part['type'] is not None:
                high_level_structure.append(cur_part)

            children = collect_children(structure, token['id'])
            cur_part = {
                'type': 'P',
                'text': [token['lemma']] + [tokens[i]['lemma'] for i in children],
            }
            high_level_structure.append(cur_part)

            used.add(token['id'])
            used |= children

            cur_part = {
                'type': None,
                'text': [],
            }

        elif token['dep'] == 'acl':
            if cur_part['type'] != 'participial' and cur_part['type'] is not None:
                high_level_structure.append(cur_part)
            children = collect_children(structure, token['id'])
            cur_part = {
                'type': 'p',
                'text': [token['lemma']] + [tokens[i]['lemma'] for i in children],
            }
            high_level_structure.append(cur_part)
            used.add(token['id'])
            used |= children
            cur_part = {
                'type': None,
                'text': [],
            }
        else:
            if cur_part['type'] is None:
                cur_part['type'] = 't'
            if token['dep'] == 'ROOT':
                cur_part['type'] = 'R'

            cur_part['text'].append(token['lemma'])
            used.add(token['id'])

    if cur_part['type'] is not None:
        high_level_structure.append(cur_part)

    return {
        'short': [x['type'] for x in high_level_structure],
        'detailed': high_level_structure,
        'hash': ''.join([x['type'] for x in high_level_structure]),
    }


def syntax_encode(sentence):
    tokens = parse(sentence).to_json()['tokens']
    cur_ids = set()
    # ищем корень
    for token in tokens:
        if token['head'] == token['id']:
            cur_ids.add(token['id'])
    while cur_ids:
        next_ids = set()
        for token_id in cur_ids:
            for token in tokens:
                if token['head'] == token_id and token['head'] != token['id']:
                    token['dep'] = tokens[token_id]['dep'] + '>' + token['dep']
                    next_ids.add(token['id'])
        cur_ids = next_ids

    return [token['dep'] for token in tokens]


def syntax_preprocessor(text):
    res = []
    for sent in sent_tokenize(text):
        res += syntax_encode(sent)
    return ' '.join(res)
