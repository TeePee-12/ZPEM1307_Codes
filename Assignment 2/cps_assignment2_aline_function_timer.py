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