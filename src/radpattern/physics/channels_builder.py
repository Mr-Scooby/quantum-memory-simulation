#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 


qs = np.array([-1,0,1]) # Spherical component
Fg = 3
Fe = 4
Fs = 4 

channel =[]


def check_excitation(F,m):
    """ Checks if the excitation/ dexcitation is posible given the F level. returns the ones possible
    return: array """
    valid_transition = []
    for q in [-1,0,1]:
        me = mg + q 
        if me > F or me < -F:
            continue
        else: 
        # Search CB coef
        valid_transition.append(me)
    retunr np.array(valid_transition)
 
def check_deexcitation(F,m):
    """ Checks if the  dexcitation is posible given the F level. returns the ones possible
    return: array """
    valid_transition = []
    for q in [-1,0,1]:
        me = mg + q 
        if me > F or me < -F:
            continue
        else: 
        # Search CB coef
        valid_transition.append(me)
    retunr np.array(valid_transition)
 

def possible_channels(Fg, Fe, Fs): 

    C  = 0 # CB product
    # Signal absorption write \Fg, mg> --> \Fe, me> = \Fe, mg +q> 
    for mg in np.arange(-Fg, Fg +1, 1): 
        valid_me = check_excitation(Fe, mg) 
        # Control absorption \Fe, me> --> \Fs, ms> = \Fs, me - q> 
        for me in valid_me: 
            valid_ms = check_deexcitation(Fs, me) 







