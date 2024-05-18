# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:06:36 2021

@author: Thomas Phelan z5349517
t.phelan@student.unsw.edu.au

ZPEM 1307 CPS Assignment 2

"""
# do not change or add any import statements
from os.path import basename
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import networkx as nx

# %%
def read(filename, n=3):
    """
    Reads a text file, splits into a  list of words then returns a list of
    tuples comtaining n number of words. Ngrams.

    Parameters
    ----------
    filename : String:
        A text file in the working directory.

    n : Int:
        Size of the n-grams to be created. The default is 3.

    Returns
    -------
    ngrams : List of tuples:
        A list of tuples of n length, consisting of individual words of
    the input file.

    """
    # file is converted into a list of words as strings
    file = open(filename).read().replace('\n', ' ').split(' ')

    # tuples of length n are genrated as items in the list "ngrams"
    return( [(tuple(file)[i: i + n]) for i in range(0 , len(file) - n)] )

# %%
def vocabulary(ngrams_list):
    """
    Flattens a nested list and returns all unique sub-list values.

    Parameters
    ----------
    ngrams_list : Nested list:
        A list of lists of n-gram tuples.

    Returns
    -------
    List of strings:
        A flat list containing all unique entries found in ngrams_list.
    """
    # inner parentheses operation flattens input
    # outer parentheses converts flattened list to a set to strip duplicates
    return( list(dict.fromkeys(
                    [item for subl in ngrams_list for item in subl] )))

# %%
def vectorize(ngrams, vocab):
    """
    Generates a numpy array of 1s and 0s that identifies if a unique
    vocabulary item features in ngrams or not.

    Parameters
    ----------
    ngrams : List of tuples(ngrams):
        A text file converted to an ngram list by the read function.
    vocab : List of tuples.
        A list of unique ngrams from a dataset as provided by the
        vocabulary function.

    Returns
    ------
    Array:
        Numpy array of 1s and zeros the same length as vocabulary.
        Features a 1 if the corresponding vocabulary item features in ngrams,
        else a 0.

    """
    # ngrams is turned into a dictionary to speed things up
    ngrams = dict.fromkeys(ngrams)

    return(np.array( [1 if i in ngrams else 0 for i in vocab] ))

# %%
def tfidf(vectors):
    """
    Applies Term Frequency * Inverse Document Frequency (tfidf) tranformation
    to a vector matrix.

    Parameters
    ----------
    vectors : Square array:
        Vector matrix of vectorized input text files.

    Returns
    -------
    Square array:
        tfidf transformed vector matrix, same shape and size as input vectors.

    """
    return( vectors * (np.log( (len(vectors))/
                      (1 + (np.sum((np.array(vectors)), axis=0)))) + 1) )

# %%
def dissimilarity(v1, v2):
    """
    This function calculates the cosine dissimilarity between two vectors of
    a vector matrix.

    Parameters
    ----------
    v1 : Array:
        A vector that exists in a TFIDF transformed vector matrix.
    v2 : Array:
        A vector that exists in the same TFIDF transformed.

    Returns
    -------
    Float:
        A number representing the dissimilarity between the two vector inputs.

    """
    return( 1 - (np.dot(v1, v2)/
                (np.sqrt (np.dot(v1, v1) * np.dot(v2, v2)))))

# %%
def distance_matrix(vectors):
    """
    Carries out a pairwise distance transformation on a matrix of vectors.

    Parameters
    ----------
    vectors : Array:
        A square vector matrix that has undergone a tfidf transformation.

    Returns
    -------
    Array:
        Pairwise distance transformed input matrix. Same and size as input.
        The dissimilarity between vectors(x,y) is at output index [x][y].

    """
    n = len(vectors)
    N = range(0, n)

    # Builds a list of dissimilarities, splits then stacks into a square matrix
    return( np.vstack(np.split((np.array
                             ( [dissimilarity (vectors[x], vectors[y])
                                for x in N for y in N] )), n)) )

# %%
def nearest_pair(distances):
    """
    Returns the two nearest vectors of a matrix using the neighbour-joining
    algorithm [Saitou and Nei, 1987].

    Parameters
    ----------
    distances : Array.
        A distance transformed vector matrix.

    Returns
    -------
    Nearest: Array.
        An array containing the indexes of the nearest vectors of the
        distances matrix.

    """
    def Q_value(d, x, y):
        '''
        Using the neighbour-joining algorithm, applies the Q matrix
        calculation to one item in a vector matrix.

        This function should be iterated through for every item in d, and the
        smallest Q value corresponds to the nearest pair.

        Parameters
        ----------
        distances : Array.
        A distance transformed vector matrix.
        x : Int:
            x-axis index of the matrix item to be calculated.
        y : Int:
            y-axis index of the matrix item to be calculated.

        Returns
        -------
        Int:
            The value at (x,y) in the distance matrix after calculation.

        '''
        return(((n-2) * d[x][y]) - np.sum(d[x]) - np.sum(d[y]) if x!=y else 0)

    n = len(distances)
    N = range(0, n)

    # Builds a list of Q Values, splits, then stacks into a square matrix
    Q_matrix = np.vstack(np.split((np.array(
              [Q_value(distances, x, y) for x in N for y in N] )), n))

    # The first pair of co-ordinates with the smallest Q Value are returned
    return( np.matrix.flatten(
           (np.array(np.where(Q_matrix == (np.min(Q_matrix))))) [: , : 1]) )

# %%
def merge_pair(graph, linkage, pair_distances, ix, pair):
    """
    Joins the two nearest nodes to make a new node.
    Takes the indexes of the nearest pair and the distances from the nearest
    pair to their parent node as defined by update_distances_matrix.
    linkage, graph and ix data sets are updated.

    Parameters
    ----------
    graph : NetworkX Graph:
        Used to hold hierarchical data from each merge step. Edges are added.

    linkage : List of lists:
        Contains hierarchical linkage data for nodes as
            [child1, child2, distance_to_leaves, number_of_leaves].
    pair_distances : Tuple:
        Distance from nodes of the nearest pair nearest to their parent node.
    ix : List:
        List of indexes of all vectors and nodes not yet merged.
        Merged nodes are removed and the new node is appended.
    pair : Tuple:
        An array of two elements that contains the indexes of the two closest
        vectors in the current distance matrix.

    Returns
    -------
    graph : NetworkX Graph:
        Updated with new edge data for the merged pair.
    linkage : List of lists:
        The input linkage with a new entry for the new node created.
    ix : List:
        Indexes of all vectors and nodes not yet merged.
        Merged nodes are removed and the new node is appended.

    """
    # Current location of merged nodes and hierarchical data held in variables
    child_1 = ix[pair[0]]
    child_2 = ix[pair[1]]
    A = linkage[child_1]
    B = linkage[child_2]

    distance_to_leaves = max(
        ( (A[2] + sum(pair_distances) + B[2]) / 2) , A[2] , B[2] )
    number_of_leaves = A[3] + B[3]

    # New node data updated in linkage, ix and graph
    linkage.append( [child_1, child_2, distance_to_leaves, number_of_leaves] )
    ix.remove(child_2)
    ix.remove(child_1)
    ix.append(len(linkage)-1)
    graph.add_edge( ix[-1], child_1, length = pair_distances[0] )
    graph.add_edge( ix[-1], child_2, length = pair_distances[1] )

    return(graph, linkage, ix)

# %%
def update_distance_matrix(distances, pair):
    """
    Takes a distance matrix and its nearest pair and updates the
    matrix according to the neighbour joining algorithm
    [Saitou and Nei, 1987].

    Parameters
    ----------
    distances : Array:
        Distacne matrix to be updated.
    pair : Tuple:
        Co ordinates of the nearest pair of vectors in the distances matrix.

    Returns
    -------
    Branches: Tuple:
        Contains the distances from each of the nearest pair nodes and the
        new node that has been created between them.
    d: Array:
        The nearest pair of vectors has been removed from the input distance
        matrix and a new vector has been added.

    """
    def update(d, i):
        '''
        Carries out the calculation between the new node and existing nodes.
        Called upon iterativley to build a new row in the distance matrix.

        Parameters
        ----------
        d : Array:
            Distacne matrix being updated.
        i : Int:
            Position of item in new row

        Returns
        -------
        Float:
            The value to be appended to the new row.

        '''
        return( (d[a][i] + d[b][i] - d[a][b]) / 2 )

    a = pair[0]
    b = pair[1]
    d = distances

    # iterate through update to build the row to be added to the matrix
    new_row=( [update(d, i) for i in range(0, len(d)) if i not in pair] )
    new_row.append(0)

    # distances from merged pair of nodes to the new node
    if len(distances) > 2:
        branch_a = ( (0.5 * d[a][b]) + (
                   (np.sum(d[a]) - np.sum(d[b])) / (2 * ( (len(d)) - 2) )))
        branches = (branch_a, d[a][b] - branch_a)
    else:
        branches = ( (0.5 * d[a][b]), (0.5 * d[a][b]) )

    # distance matrix is updated with new data
    d=np.delete(d, (pair), 1)
    d=np.delete(d, (pair), 0)
    d=np.pad(d, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    d[   -1]=new_row
    d[:, -1]=new_row

    return(branches, d)


# %%
def cluster(vectors):
    """
    Takes vectorized distance matrix, iterates through a
    nearest-neighbour algorithm to merge nodes and update the matrix
    until a clustered dataset of merged nodes is returned as a tree.

    Parameters
    ----------
    vectors : 2d numpy array.
        A tfidf transformed distance vector matrix.

    Returns
    -------
    Linkage: List of lists:
        Contains heirarchical clustering data of all nodes.
    graph : NetworkX Graph:
        Graph edge data contains branch length data from each merge as edges.

    """
    n = len(vectors)
    distances = distance_matrix(vectors)
    graph = nx.Graph()
    linkage = ([ [None, None, 0, 1] ]*n)
    ix = [* range(0,  n) ]

    while len(distances) > 1:
        pair = nearest_pair (distances)
        pair_distances, distances =(
            update_distance_matrix (distances, pair) )
        graph , linkage , ix =(
            merge_pair(graph, linkage, pair_distances, ix, pair) )
        
    del linkage[0: n]

    return(np.array(linkage), graph)


# %%
# Do not delete or change code beneath this line.
def to_newick(filename, graph, labels=None):
    """
    Save a graph in Newick format, so that it can be read in by fancy
    dendrogram plotting applications like https://itol.embl.de/.

    The graph is saved rooted at its maximal node, so node labels must be
    integers.

    Parameters
    ----------
    filename : str
        File into which tree should be saved. Should have ".tree" suffix.
    graph : networkx.Graph
        Graph to save. Graph nodes should be integer-labelled.
    labels : TYPE, optional
        Leaf labels, indexed by type node labels. The default is None.

    """
    def newick_string(node, visited=None):
        if not visited:
            visited = set()
        visited.add(node)
        if len(graph[node]) == 1:
            if labels:
                return labels[node]
            else:
                return str(node)
        strgen = (newick_string(n, visited) +
                  ':' + str(graph[node][n]['length'])
                  for n in graph[node] if n not in visited)
        return '(' + ','.join(strgen) + ')'
    with open(filename, 'w') as file:
        file.write(newick_string(max(graph.nodes)))


def main():
    """
    Run a clustering analysis on the assignments in the "assignments"
    directory, which should be in your current working directory.

    Displays the resulting heirarchy as a dendrogram and saves the
    neighbour-joining tree in "tree.tree".

    Raises
    ------
    RuntimeError
        Raised if it can't find the assignments.

    Returns
    -------
    fig : a matplotlib figure
        A dendrogram of the heirarchy.

    """
    # read in the assignments
    data = []
    labels = []
    files = list(glob('assignments/*'))
    if not files:
        raise RuntimeError("I can't find the assignments. "
                           "Are they in your current working directory?")
    # Uncomment the following line to test on a subset of assignments
    # files = np.random.choice(glob('assignments/*'), size=50, replace=False)
    for filename in files:
        data.append(read(filename))
        labels.append(basename(filename)[:-4])

    # extract the vocab and convert assignments to vectors
    vocab = vocabulary(data)
    vectors = [vectorize(d, vocab) for d in data]

    # transform the features to make rare features more important
    vectors = tfidf(vectors)

    # cluster the assignments
    linkage, graph = cluster(vectors)

    # save the tree for prettier plotting
    to_newick('tree.tree', graph, labels)

    # plot the result
    fig = plt.figure(figsize=(10, 10))
    dendrogram(linkage, labels=labels, orientation='left')
    return fig


# %%
# Do not delete or change this code
if __name__ == '__main__':
    main()
