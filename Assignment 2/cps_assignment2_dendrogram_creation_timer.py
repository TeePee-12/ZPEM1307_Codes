# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:06:36 2021

@author: Thomas Phelan z5349517

a_line CPS timer test program (time for dendrogram creation)

Save this file in the same directory as aline

This will call the mainfunction in aline, 
return the dendrogram, and print the time taken

"""

import time
import aline
start=time.perf_counter()
aline.main()
end=time.perf_counter()
print('dendrogram generated in ',end-start,' seconds')
