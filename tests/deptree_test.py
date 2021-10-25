from collections import defaultdict

from src.deptree import DepTree, build_tree_from_map


def test_build_tree_from_map():
    test_tree_map = defaultdict(list)
    test_tree_map.update(
        {
            1: [
                DepTree.Node({'id': 2, 'lemma': 'b', 'pos': 'foo', 'dep': 'bar'}),
                DepTree.Node({'id': 3, 'lemma': 'c', 'pos': 'foo', 'dep': 'bar'}),
            ],
            2: [
                DepTree.Node({'id': 4, 'lemma': 'd', 'pos': 'foo', 'dep': 'bar'}),
            ]
        },
    )
    test_root = DepTree.Node({'id': 1, 'lemma': 'a', 'pos': 'foo', 'dep': 'bar'})

    expected_tree = DepTree.Node({'id': 1, 'lemma': 'a', 'pos': 'foo', 'dep': 'bar'})
    expected_tree.children = [
        DepTree.Node({'id': 2, 'lemma': 'b', 'pos': 'foo', 'dep': 'bar'}),
        DepTree.Node({'id': 3, 'lemma': 'c', 'pos': 'foo', 'dep': 'bar'}),
    ]
    expected_tree.children[0].children = [
        DepTree.Node({'id': 4, 'lemma': 'd', 'pos': 'foo', 'dep': 'bar'}),
    ]

    assert build_tree_from_map(test_root, test_tree_map) == expected_tree
