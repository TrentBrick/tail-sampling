# Sampling functions

import numpy as np
import enchant
from decodeLogits import *

def nucleus_calc(vals, thresh ):
    return np.argmax(np.cumsum(vals)>thresh)+1

def new_tfs(second, thresh):
    only_pos = np.abs(second)
    sec_weights = only_pos/only_pos.sum()
    tail_id = np.argmax(np.cumsum(sec_weights)>thresh)+1
    return tail_id
        
def flat(sps, p):
    return sps.shape[0]-np.argmax(np.flip(sps)>p)+1

def remove_non_words(tokens):

    real_words = []
    d = enchant.Dict("en_US")
    for t in tokens: 
        t = decoder_text([t])
        t = t.strip()
        if t == '':
            continue
        if len(t)==1:
            up = t.upper()
            if up!='I' and up!='A':
                continue
        if d.check(t):
            real_words.append(t)
    return real_words
    
def get_specific_positions(tokens, starting_pos):
    if len(tokens)< 3:
        return ([None, None, None], [0,0,0])
    l = len(tokens)
    mid = l//2
    return ([tokens[0], tokens[mid] ,tokens[-1]], [starting_pos, starting_pos+mid , starting_pos+l])