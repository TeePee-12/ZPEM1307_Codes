# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:34:27 2021

@author: Thomas
"""

def alternate(x, y):
    """
    Takes two lists of equal length as input and returns a single output list
    containgin alternating list items of the inputs.
    Input lists of different lengths will not return a useful output.
    Parameters
    ----------
    x : A lsit containg any numebr of items.
    y : A list the same length of x.
    Returns
    -------
    An alternating list containg even items of x and odd items of y .
    """
    x_y_alternate=[range(0,(len(y)))]
    if len(x) != len(y):
        return('error: only accepts lists of equal length')
    else:
        for i in range(len(x)):
            if i % 2 == 0:
                x_y_alternate[i]=x[i]
            else:
                x_y_alternate[i]=y[i]    
    return(x_y_alternate)

print(alternate([1,0],[1,1]))