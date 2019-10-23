# Sampling functions

import numpy as np
import enchant
from decodeLogits import *
import matplotlib.pyplot as plt

def nucleus_calc(vals, thresh ):
    return np.argmax(np.cumsum(vals)>thresh)+1

def new_tfs(second, thresh):
    only_pos = np.abs(second)
    sec_weights = only_pos/only_pos.sum()
    tail_id = np.argmax(np.cumsum(sec_weights)>thresh)+1
    return tail_id
        
def flat(sps, p):
    return sps.shape[0]-np.argmax(np.flip(sps)>p)+1

def remove_non_words(tokens, wantPrint=False):

    real_words = []
    abs_positions = []
    d = enchant.Dict("en_US")
    for ind, t in enumerate(tokens): 
        t = decoder_text([t])
        t = t.strip()
        if wantPrint:
            print(t)
        if t == '':
            continue
        if len(t)==1:
            up = t.upper()
            if up!='I' and up!='A':
                continue
        if d.check(t):
            real_words.append(t)
            abs_positions.append(ind)
    return [real_words, abs_positions]
    
def get_specific_positions_from_probs(real_word_probs, prob_slices_wanted, rand_range=2):
    word_inds = []
    sel_word_probs = []
    for ind, p in enumerate(prob_slices_wanted): 
        
        nearest_val, p_nearest_ind = find_nearest(real_word_probs, p)
        
        real_word_probs_clone = np.array(real_word_probs)
        
        while p_nearest_ind in word_inds: 
            #need to search for something that is closer. 
            real_word_probs_clone[p_nearest_ind] = 100000000 
            nearest_val, p_nearest_ind = find_nearest(real_word_probs_clone, p)
            #print(p_nearest_ind)
            
            
            7
[' Charles' 'steadily' 'then' 'poured' 'tried' 'kept' 'held']
[0.10374737 0.02087543 0.01230764 0.00359681 0.01044577 0.00544149]
            
        #print(p_nearest_ind)
        
        if p_nearest_ind+rand_range > len(real_word_probs):
            # it will find the very last value so sample only 5 ahead for this one. 
            sel_word_ind = np.random.choice( np.arange(p_nearest_ind-(2*rand_range), p_nearest_ind) ,1)[0]
            word_inds.append(sel_word_ind)
            sel_word_probs.append(real_word_probs[sel_word_ind])
            
        else: 
            word_inds.append(p_nearest_ind)
            sel_word_probs.append(nearest_val)
            
    return word_inds, sel_word_probs
    
    
def find_nearest(array, value):
    # taken from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array 
    # returns the actual number and the index. 
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
    
def get_specific_positions(tokens, starting_pos, rel_pos):
    if len(tokens)< 3:
        return ([None, None, None], [0,0,0])
    l = len(tokens)
    mid = l//2
    return ([tokens[0], tokens[mid] ,tokens[-1]],  [starting_pos+rel_pos[0], starting_pos+rel_pos[mid] ,starting_pos+rel_pos[-1]])

def bar_plot_columns(df, title_app):
    x_vals = range(df.shape[1])
    plt.bar(x_vals, df.mean())
    plt.title('Mean Replacability Score - '+title_app)
    plt.xticks(x_vals, df.columns, rotation='vertical')
    plt.show()
    
def abs_diff_of_different_locations(df, t_lab, n_lab):
    for i in range(1, 4):
        print('position:',i, '=',np.abs(df[t_lab+'-'+str(i)].mean() - df[n_lab+'-'+str(i)].mean()))