from collections import defaultdict
import pytest

from src.deptree import DepTree, build_tree_from_map


@pytest.fixture
def sample_tree():
    tree = DepTree.Node({'id': 1, 'lemma': 'a', 'pos': 'foo', 'dep': 'a_bar'})
    tree.children = [
        DepTree.Node({'id': 2, 'lemma': 'b', 'pos': 'foo', 'dep': 'b_bar'}),
        DepTree.Node({'id': 3, 'lemma': 'c', 'pos': 'foo', 'dep': 'c_bar'}),
    ]
    tree.children[0].children = [
        DepTree.Node({'id': 4, 'lemma': 'd', 'pos': 'foo', 'dep': 'd_bar'}),
    ]
    return tree


def test_build_tree_from_map(sample_tree):
    test_tree_map = defaultdict(list)
    test_tree_map.update(
        {
            1: [
                DepTree.Node({'id': 2, 'lemma': 'b', 'pos': 'foo', 'dep': 'b_bar'}),
                DepTree.Node({'id': 3, 'lemma': 'c', 'pos': 'foo', 'dep': 'c_bar'}),
            ],
            2: [
                DepTree.Node({'id': 4, 'lemma': 'd', 'pos': 'foo', 'dep': 'd_bar'}),
            ]
        },
    )
    test_root = DepTree.Node({'id': 1, 'lemma': 'a', 'pos': 'foo', 'dep': 'a_bar'})

    assert build_tree_from_map(test_root, test_tree_map) == sample_tree


def test_deptree_str(sample_tree):
    assert str(sample_tree) == 'a_bar(b_bar(d_bar) c_bar)'
