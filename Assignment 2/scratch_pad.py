# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:13:58 2021

@author: Thomas

"""

import time
import aline
start=time.perf_counter()
aline.main()
end=time.perf_counter()
print('dendrogram generated in ',end-start,' seconds')
#%%
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:13:58 2021

@author: Thomas Phelan z5349517

A_line CPS timer test program

Save this file in the same directory as aline

This code will run most of the main function for only the first iteration.
This code returns runtimes for each functions you built in your assignment.


"""

import time
from glob import glob
import networkx as nx
import aline 


start_time = time.perf_counter()

data = []

files = list(glob('assignments/*'))
#files = np.random.choice(glob('assignments/*'), size=50, replace=False)
                         
for filename in files:
    data.append(aline.read(filename))


read_time = time.perf_counter()
print('data read in ',(read_time - start_time),' seconds')
print(' ')


vocab = aline.vocabulary(data)
vocab_time = time.perf_counter()
print('vocab built in ',(vocab_time - read_time),' seconds')
print('Total runtime ',(vocab_time - start_time),' seconds')
print(' ')


vectors = [aline.vectorize(d, vocab) for d in data] 
vectors_time = time.perf_counter()
print('vectorized in ',(vectors_time -vocab_time),' seconds')
print('Total runtime ',(vectors_time - start_time),' seconds')
print(' ')


vectors = aline.tfidf(vectors)
tfidf_time = time.perf_counter()
print('TFIDF transformed in ',(tfidf_time - vectors_time),' seconds')
print('Total runtime ',(tfidf_time - start_time),' seconds')
print(' ')


distances = aline.distance_matrix(vectors)
distances_time = time.perf_counter()
print('distances_matrix created in ',(distances_time - tfidf_time),' seconds')
print('Total runtime ',(distances_time - start_time),' seconds')
print(' ')


graph = nx.Graph()
linkage = ([[None,None,0,1]]*len(vectors))
ix = [*range(0, len(vectors))]
cluster_dataset_time = time.perf_counter()
print('cluster data setup in ',(cluster_dataset_time - distances_time),' seconds')
print('Total runtime ',(cluster_dataset_time - start_time),' seconds')
print(' ') 


pair = (aline.nearest_pair(distances))
nearest_pair_time = time.perf_counter()
print('nearest pair found in ',(nearest_pair_time - cluster_dataset_time),' seconds')
print('Total runtime ',(nearest_pair_time - start_time),' seconds')
print(' ') 

  
pair_distances, distances =(aline.update_distance_matrix(distances,pair))
update_distance_matrix_time = time.perf_counter()
print('distance matrix updated in ',(update_distance_matrix_time - nearest_pair_time),' seconds')
print('Total runtime ',(update_distance_matrix_time - start_time),' seconds')
print(' ')   

    
graph , linkage , ix =(aline.merge_pair(graph , linkage , pair_distances , ix , pair))
merge_pair_time = time.perf_counter()
print('merge pair completed in ',(merge_pair_time - update_distance_matrix_time),' seconds')
print('Total runtime ',(merge_pair_time - start_time),' seconds')
print(' ')

#%%
from glob import glob
import numpy as np
import aline
data = []
files = list(glob('assignments/*'))
#files = np.random.choice(glob('assignments/*'), size=50, replace=False)
                         
for filename in files:
    data.append(aline.read(filename))

vocab = aline.vocabulary(data)
ngrams = vocab
vocab = np.array(vocab)
ngrams = np.array(ngrams)
features = (np.array(vocab) == (np.array(ngrams)))


#%%

import time
from os.path import basename
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import networkx as nx
import aline 

tic = time.perf_counter()

data = []
files = list(glob('assignments/*'))
#files = np.random.choice(glob('assignments/*'), size=50, replace=False)
                         
for filename in files:
    data.append(aline.read(filename))
    
vocab = aline.vocabulary(data)

vectors = [aline.vectorize(d, vocab) for d in data] 

vectors = aline.tfidf(vectors)

linkage, graph = aline.cluster(vectors)
    
    # save the tree for prettier plotting
   # to_newick('tree.tree', graph, labels)
    
    # plot the result
   # fig = plt.figure(figsize=(10, 10))
    #dendrogram(linkage, labels=labels, orientation='left')
    #return fig
    
    
toc = time.perf_counter()
print('code compiled in ', round((toc-tic),2), ' seconds')

#%%
import numpy as np
import aline

vectors = np.random.rand(200,200)

distances = aline.distance_matrix(vectors)    

Q = aline.nearest_pair(distances)

#%%
import numpy as np
import aline
distances = np.array([[0, 0.5, 0.18350342],
                      [0.5, 0, 0.18350342],
                      [0.18350342, 0.18350342, 0]])

D=distances
n=len(D)
N=range(0,n)
neighbours = lambda x,y: (((n-2)*D[x][y])-np.sum(D[x])-np.sum(D[y]) if x!=y else 0)
Q=np.vstack(np.split((np.array([neighbours(x,y) for x in N for y in N])),n))
print(Q)
pair=np.matrix.flatten((np.array(np.where(Q==(np.min(Q)))))[:,:1])
print(pair)

#%%
'''
update distance matrix
test case as pulled from wikipedia neighbour joining article
'''

import numpy as np
import aline
distances = np.array([[1, 1, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [1, 0, 1, 1, 1],
                      [1, 0, 1, 0, 1],
                      [1, 1, 1, 1, 1]])

pair = (0, 3)


#distances = np.array([[0, 0.3, 0.4],
#                      [0.3, 0, 0.2],
#                      [0.4, 0.2, 0]])
#pair = (1, 2)



a = pair[0]
b = pair[1]
d = distances

update = lambda x: ((d[a][x]+d[b][x]-d[a][b])/2)

new_row=[update(i) for i in range(0, len(d)) if i not in pair]
new_row.append(0)

branch_a = ((0.5*d[a][b])+((np.sum(d[a])-np.sum(d[b]))/(2*((len(d))-2))))
branches = (branch_a , d[a][b]-branch_a)
print('branches')
print(branches)
print(' ')

print('new row is')
print(new_row)
print(' ')

print('distances before removal')
print(d)
print('')



d=np.delete(d,(pair),1)
d=np.delete(d,(pair),0)

print('distances after removal')
print(d)
print('')    

d=np.pad(d, [(0, 1), (0, 1)], mode='constant', constant_values=0,)

print('padded distances is')
print(d)
print(' ')

d[-1]=new_row
d[:,-1]=new_row

print('final distance vector is')
print(d)
print(' ')
#return(update)
#%%

import aline
import numpy as np
#distances = np.array([[0,5,9,9,8],
#                     [9,10,0,8,7],
#                      [9,10,8,0,3],
#                      [8,9,7,3,0]])
#pair = (0, 1)

distances = np.array([[0, 0.3, 0.4],
                      [0.3, 0, 0.2],
                      [0.4, 0.2, 0]])
pair = (1, 2)

outdisst, outbranch = (aline.update_distance_matrix(distances,pair))
    
    
#%%
tic1 = time.perf_counter()
ngrams=aline.read('assignments\zubat.txt')

data = []
   
files = list(glob('assignments/*'))
for filename in files:
    data.append(aline.read(filename))

vocabulary = aline.vocabulary(data)

tic2 = time.perf_counter()

vectors=aline.vectorize(ngrams, vocabulary)

#ngramdict=dict.fromkeys(ngrams,0)
#for i in ngramdict:
#   [1 for i in vocabulary if i in ngramdict]  
tic3 = time.perf_counter()   

print((tic3-tic1),(tic3-tic2))

#%% dont_run_this_cell

#runcell('dont_run_this_cell')
import aline
rick_astley = aline.read('rick.txt',5)

print(rick_astley[43])
print(rick_astley[48])
print(rick_astley[53])
print(rick_astley[58][0:2])
print(rick_astley[60]) 
print(rick_astley[65])
print(rick_astley[70])
print(rick_astley[75][0:3])

#%%

"""
Merge Pair Test
"""

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
    
    
#%%
import aline
import networkx as nx
from glob import glob
import numpy as np

data = []
files = list(glob('assignments/*'))
#files = np.random.choice(glob('assignments/*'), size=50, replace=False)
for filename in files:data.append(aline.read(filename))
    
vocab = aline.vocabulary(data)

vectors = [aline.vectorize(d, vocab) for d in data] 

vectors = aline.tfidf(vectors)


n = len(vectors)
distances = aline.distance_matrix(vectors)
graph = nx.Graph()
linkage = [[(),(),0,1]]*n
ix = [*range(0, n)]

print(ix)
print("linkage",len(linkage))
print('start')   
while len(distances)>1:
    pair = aline.nearest_pair(distances)   
    pair_distances, distances =(
        aline.update_distance_matrix(distances,pair))
    print(pair)   
    graph , linkage , ix =(
        aline.merge_pair(graph , linkage , pair_distances , ix , pair))
    print(ix)
    print(distances)


#%%

pair = (0, 1)
pair_distances = (0.5, 0.4)
linkage = [[None, None, 0, 1],
           [None, None, 0, 1],
           [None, None, 0, 1],
           [0, 1, 0.45, 2]]
ix = [2, 3]

child_1 = ix[pair[0]]
child_2 = ix[pair[1]]
    
A = linkage[child_1]
B = linkage[child_2]
    
distance_to_leaves = max(
    ((A[2]+sum(pair_distances)+B[2])/2) , A[2] , B[2] )  

number_of_leaves = A[3]+B[3]

linkage.append([ child_1 , child_2 , distance_to_leaves , number_of_leaves ])
        
ix.remove(child_2)
ix.remove(child_1)     
ix.append(len(linkage)-1)
    


#%%
import numpy as np
import aline
import networkx as nx

vectors = np.array([[0, 1, 0],
                    [0, 1, 1],
                    [1, 1, 1]])


n = len(vectors)
distances = aline.distance_matrix(vectors)
graph = nx.Graph()
linkage = [[None,None,0,1]]*n
ix = [*range(0, n)]
    
while len(distances) > 1:
    pair = aline.nearest_pair(distances)
       
    pair_distances, distances =(
        aline.update_distance_matrix(distances,pair))
       
    graph , linkage , ix =(
        aline.merge_pair(graph , linkage , pair_distances , ix , pair))
    
#del linkage[0:n]

print(np.array(linkage))

#%%
from os.path import basename
from glob import glob
import aline
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import networkx as nx
vectors = np.array([[1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [1, 0, 0, 1]])
n = len(vectors)
distances = aline.distance_matrix(vectors)
graph = nx.Graph()
linkage = ([ [None, None, 0, 1] ]*n)
ix = [* range(0,  n) ]

while len(distances) > 1:

    # nearest pair is found
    pair = aline.nearest_pair(distances)
    print(pair)
    # distance matrix updated, distances from pair to node calculated
    pair_distances, distances =(
        aline.update_distance_matrix (distances, pair) )
    print(pair_distances, distances)
    # data structures updated to hold hierarchical data of merges
    graph , linkage , ix =(
        aline.merge_pair(graph, linkage, pair_distances, ix, pair) )
    print(linkage, ix, graph)
    
    print(' ')
    



