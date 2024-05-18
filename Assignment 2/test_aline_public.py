# -*- coding: utf-8 -*-
"""
@author: Ben Kaehler
"""
import numpy as np
from scipy.cluster.hierarchy import is_valid_linkage
import networkx as nx

import aline

def test_read():
    "Test crucial aspects of assignment reading"
    expected_first = ('fright', 'silk')
    expected_last = ('dealing', 'stirring')
    expected_overline = ('mile', 'witness')
    observed = aline.read('assignments/charmander.txt', 2)
    assert observed[0] == expected_first, 'problem with first ngram'
    assert observed[-1] == expected_last, 'problem with last ngram'
    assert expected_overline in observed, 'problem with line breaks'

def test_vocabulary():
    "Test that vocab appropriately combines documents"
    ngram_list = [[('a', 'b'), ('c', 'd')],
                  [('c', 'd'), ('e', 'f')]]
    expected = {('a', 'b'), ('c', 'd'), ('e', 'f')}
    observed = aline.vocabulary(ngram_list)
    assert isinstance(observed, list), 'vocab is the wrong type'
    assert set(observed) == expected, 'vocab has the wrong elements'
    
def test_vectorize():
    "Test document feature extraction"
    ngrams = [('a', 'b'), ('c', 'd')]
    vocab = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    expected = [1, 1, 0]
    observed = aline.vectorize(ngrams, vocab)
    assert (observed == expected).all(), 'vector should be [1, 1, 0]'
    
def test_tfidf():
    "Test TFIDF"
    vectors = [np.array([1,1,0]), np.array([0,1,1])]
    expected = [[1, 0.59453489, 0],
                [0, 0.59453489, 1]]
    observed = aline.tfidf(vectors)
    np.testing.assert_allclose(observed, expected)

def test_dissimilarity():
    "Trivial test of dissimilarity measure"
    v1 = np.array([1, 0, 1])
    v2 = np.array([1, 0, 1])
    expected = 0
    observed = aline.dissimilarity(v1, v2)
    assert abs(observed - expected) < 1e-9, 'non-zero dissimilarity'
    
def test_distance_matrix():
    "Test that a distance matrix is appropriately generated"
    vectors = np.array([[1,1,0],[0,1,1],[1,1,1]])
    expected = np.array([[0, 0.5, 0.18350342],
                         [0.5, 0, 0.18350342],
                         [0.18350342, 0.18350342, 0]])
    observed = aline.distance_matrix(vectors)
    assert (abs(observed - expected) < 1e-9).all(), 'distance matrix wrong'

def test_nearest_pair():
    "Test neighbour-joining nearest pair on a triple"
    distances = np.array([[0, 0.5, 0.18350342],
                          [0.5, 0, 0.18350342],
                          [0.18350342, 0.18350342, 0]])
    observed = aline.nearest_pair(distances)
    assert observed[0] in (0, 1, 2), 'first pair element not an index'
    assert observed[1] in (0, 1, 2), 'first pair element not an index'
    assert observed[0] != observed[1], 'pair does not describe two elements'

def test_merge_pair():
    "Test neighbour-joining merge step"
    pair = (0, 1)
    linkage = [[None, None, 0, 1]]*3
    pair_distances = (0.5, 0.4)
    ix = [0, 1, 2]
    expected_linkage = [[None, None, 0, 1],
                        [None, None, 0, 1],
                        [None, None, 0, 1],
                        [0, 1, 0.45, 2]]
    expected_ix = [2, 3]
    graph = nx.Graph()
    aline.merge_pair(graph, linkage, pair_distances, ix, pair)
    assert linkage[:-1] == expected_linkage[:-1], 'start of linkage changed'
    np.testing.assert_allclose(linkage[-1], expected_linkage[-1])
    assert ix == expected_ix, 'ix not updated correctly'
    np.testing.assert_allclose(graph[3][0]['length'], pair_distances[0])
    np.testing.assert_allclose(graph[3][1]['length'], pair_distances[1])
    
def test_update_distance_matrix():
    "Test neighbour-joining distance matrix update step"
    distances = np.array([[0, 0.3, 0.4],
                          [0.3, 0, 0.2],
                          [0.4, 0.2, 0]])
    pair = (1, 2)
    expected_distances = np.array([[0, 0.25],
                                   [0.25, 0]])
    expected_ds = (0.05, 0.15)
    observed_ds, observed_distances = aline.update_distance_matrix(distances,
                                                                   pair)
    assert (abs(observed_distances - expected_distances) < 1e-9).all(), \
        'distances incorrect'
    np.testing.assert_allclose(observed_ds, expected_ds)

def test_cluster():
    "Test neighbour-joining algorithm"
    vectors = np.array([[0, 1, 0],
                        [0, 1, 1],
                        [1, 1, 1]])
    expected_lengths = [(0, 1, 0.29289321881345254),
                        (1, 2, 0.18350341907227408),
                        (2, 0, 0.42264973081037416)]
    observed_linkage, observed_graph = aline.cluster(vectors)
    assert observed_linkage.shape == (2, 4), 'linkage should have two rows'
    assert 3 in observed_linkage[1,:2], \
        'second merge must connect to first cluster'
    assert list(observed_linkage[:,-1]) == [2, 3], \
        'last column must count leaves'
    is_valid_linkage(observed_linkage, throw=True)
    for s, t, d in expected_lengths:
        obs_d = nx.shortest_path_length(observed_graph, s, t, weight='length')
        np.testing.assert_allclose(obs_d, d)
        
