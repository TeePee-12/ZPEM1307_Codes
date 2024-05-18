# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:06:36 2021

@author: Thomas Phelan z5349517
t.phelan@student.unsw.edu.au

ZPEM 1307 CPS Assignment 2 

"""

import numpy as np
from scipy.cluster.hierarchy import is_valid_linkage
import networkx as nx
import aline


def test_read():
    "Uses two assignments as test cases to test all aspects of read"
    expected_first = ('affairs', 'silk', 'pindarus')
    expected_last = ('alexandria', 'alb', 'alb')
    expected_overline = ('alb', 'alb', 'kisses')
    observed = aline.read('assignments/zubat.txt')
    assert observed[0] == expected_first, 'problem with first ngram'
    assert observed[-1] == expected_last, 'problem with last ngram'
    assert expected_overline in observed, 'problem with line breaks'
    
    expected_first = ('fright', 'silk', 'pindarus', 'every', 'gentleness')
    expected_last = ('crave', 'williams', 'alexandria', 'forest', 'forest')
    expected_overline = ('crave', 'williams', 'alexandria', 'forest', 'forest')
    observed = aline.read('assignments/snorlax.txt', 5)
    assert expected_overline in observed, 'problem with line breaks'
    assert observed[0] == expected_first, 'problem with first ngram'
    assert observed[-1] == expected_last, 'problem with last ngram'

def test_vocabulary():
    "Test that vocab builds a list of unique entries"
    ngram_list = [[('never', 'gonna'), ('give', 'you', 'up')],
                  [('never', 'gonna'), ('let', 'you', 'down')]]
    expected = {('never', 'gonna'), ('give', 'you', 'up'), 
                ('let', 'you', 'down')}
    observed = aline.vocabulary(ngram_list)
    assert isinstance(observed, list), 'vocab is the wrong type'
    assert set(observed) == expected, 'vocab has the wrong elements'
    
def test_vectorize():
    "Test document feature extraction"
    ngrams = [('never', 'gonna'), ('give', 'you', 'up')]
    vocab  = [('never', 'gonna'), ('give', 'you', 'up'), 
              ('let', 'you', 'down')]
    expected = [1, 1, 0]
    observed = aline.vectorize(ngrams, vocab)
    assert (observed == expected).all(), 'vector should be [1, 1, 0]'  
    
def test_tfidf():
    "Test calculation of tfidf"
    vectors = np.array([[1, 0, 0],
                        [1, 1, 1],
                        [1, 0, 1]])
    expected = ([[0.712318,        0, 0],
                 [0.712318, 1.405465, 1],
                 [0.712318,        0, 1]])
    observed = aline.tfidf(vectors)
    np.testing.assert_allclose(observed, expected, rtol=2e-7), 'problem with tfidf'
       
    
def test_dissimilarity():
    "Exhasutive test of dissimilarity function"
    v1 = np.array([1, 0, 1])
    v2 = np.array([1, 0, 1])
    expected = 0
    observed = aline.dissimilarity(v1, v2)
    assert abs(observed - expected) < 1e-9, 'non - zero dissimilarity'  
    
    v1 = np.array([0, 1, 0])
    v2 = np.array([1, 0, 1])
    expected = 1
    observed = aline.dissimilarity(v1, v2)
    assert expected == observed, 'non - one dissimilarity' 
    
    v1 = np.array([0, 1, 1])
    v2 = np.array([1, 0, 1])
    expected = 0.5
    observed = aline.dissimilarity(v1, v2)
    assert expected == observed, 'problem with dissimilarity'
    
def test_distance_matrix():
    "Test that a distance matrix is appropriately generated"
    vectors = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1]])
    expected = np.array([[0  , 1, 0.5       ],
                         [1  , 0, 0.29289322],
                         [0.5, 0.29289322, 0]])
    observed = aline.distance_matrix(vectors)
    np.testing.assert_allclose(observed, expected), 'distance matrix wrong' 
    
def test_nearest_pair():
    "Test neighbour-joining returns only the first nearest pair in a matrix"
    distances = np.array([[1, 0, 1, 0, 1],
                          [0, 1, 0, 0, 1],
                          [1, 0, 1, 1, 1],
                          [1, 0, 1, 0, 1],
                          [1, 0, 1, 0, 1]])
    observed = aline.nearest_pair(distances)
    assert len(observed) == 2, 'Nearest_pair does not return two values'
    assert observed[0] == 0, 'first pair element not an index'
    assert observed[1] == 3, 'first pair element not an index'
    assert observed[0] != observed[1], 'pair does not describe two elements' 

def test_merge_pair():
    "Test one neighbour-joining merge step"
    pair = (1, 3)
    linkage = [[None, None, 0, 1],
               [None, None, 0, 1],
               [None, None, 0, 1],
               [None, None, 0, 1],
               [None, None, 0, 1],
               [0, 3, 0.5, 2]]
    pair_distances = (0.2, 0.3)
    ix = [ 1, 2, 4, 5]
    expected_linkage = [[None, None, 0, 1],
                        [None, None, 0, 1],
                        [None, None, 0, 1],
                        [None, None, 0, 1],
                        [None, None, 0, 1],
                        [0, 3, 0.5, 2],
                        [2, 5, 0.5, 3]]
                        
    expected_ix = [1, 4, 6]
    graph = nx.Graph()
    aline.merge_pair(graph, linkage, pair_distances, ix, pair)
    assert linkage[:-1] == expected_linkage[:-1], 'start of linkage changed'
    np.testing.assert_allclose(linkage[-1], expected_linkage[-1])
    assert ix == expected_ix, 'ix not updated correctly'
    np.testing.assert_allclose(graph[6][2]['length'], pair_distances[0])
    np.testing.assert_allclose(graph[6][5]['length'], pair_distances[1])
    
def test_update_distance_matrix():
    "Test both branches of the update_distance_matrix function"
    distances = np.array([[0, 1],
                          [1, 0]])
    pair = (0, 1)
    expected_distances = np.array([0])
    expected_ds = (0.5, 0.5)
    observed_ds, observed_distances = aline.update_distance_matrix(distances,
                                                                   pair)
    assert (abs(observed_distances - expected_distances) < 1e-9).all(), \
        'distances incorrect'
    np.testing.assert_allclose(observed_ds, expected_ds)  
    
    distances = np.array([[0, 5 , 9 , 9 , 8],
                          [5, 0 , 10, 10, 9],
                          [9, 10, 0 , 8 , 7],
                          [9, 10, 8 , 0 , 3],
                          [8, 9 , 7 , 3 , 0]])
    pair = (0, 1)
    expected_distances = np.array([[0, 8, 7, 7],
                                   [8, 0, 3, 7],
                                   [7, 3, 0, 6],
                                   [7, 7, 6, 0]])
    expected_ds = (2.0, 3.0)
    observed_ds, observed_distances = aline.update_distance_matrix(distances,
                                                                   pair)
    assert (abs(observed_distances - expected_distances) < 1e-9).all(), \
        'distances incorrect'
    np.testing.assert_allclose(observed_ds, expected_ds)    
    
def test_cluster():
    "Test neighbour-joining algorithm"
    vectors = np.array([[1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0]])
    expected_lengths = [
        (0, 3, 0.5),(1, 2, 0.0),(4, 5, 0.5)]
    observed_linkage, observed_graph = aline.cluster(vectors)
    assert observed_linkage.shape == (3, 4), 'linkage should have two rows'
    assert 5 in observed_linkage[2,:2], \
        'second merge must connect to first cluster'
    assert list(observed_linkage[:,-1]) == [2, 2, 4], \
        'last column must count leaves'
    is_valid_linkage(observed_linkage, throw=True)
    for s, t, d in expected_lengths:
        obs_d = nx.shortest_path_length(observed_graph, s, t, weight='length')
        np.testing.assert_allclose(obs_d, d)

    
    
    
    