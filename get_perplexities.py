#!/usr/bin/env python3
import fire
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import gzip
import model, perplexities_calculations, encoder

"""

Input data for which perplexities are returned at every time point. 

Also get the probability that the model assigns to the real world chosen.
"""


"""
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)     
    """


def interact_model( # some other variables are initialized below
    general_path = '',
    alpha=None,
    nuc_prob=0.25,
    sampler='tfs', #n, k or tfs
    perc_acc=0.99,
    pre_prepared_prompts = True,
    model_name='774M',
    seed=27,
    batch_size=25, # 500
    softmax_output_size = 50527,
    to_perp_length = 250, # as need to add the prompt and the completion
    prompt_length = 100, # will trim off of what is returned. 
    temperature=1,
    top_k=0,
    models_dir='../gpt-2/models',    
):
    
    # initializing some other variables ==========
    nsamples=batch_size # should equal the batch size. 
    pre_prepared_to_perplex_data_path = general_path+'Human_StoryPrompts_Completion.csv'

    experiment_name = "perplexity_scores_for_the_dataset_%s-model_%s-seed_%s" %(pre_prepared_to_perplex_data_path,model_name, seed)

    np.random.seed(seed)
    print('dataframe being loaded in',pre_prepared_to_perplex_data_path)
    df=pd.read_csv(pre_prepared_to_perplex_data_path)
 
    start = np.arange(0,df.shape[0],batch_size)
    end = start + batch_size
    #===========================

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    generated_length=150

    if generated_length is None:
        generated_length = hparams.n_ctx // 2
    elif generated_length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    
    #saving all of the perplexities from the different batches taken in:
    all_perplexities = []
    all_logits = []
    all_text = []

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [None, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = perplexities_calculations.perp_calc(
            hparams=hparams, length=generated_length,
            context=context,
            batch_size=batch_size)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # CHANGE THIS TO BE IN RANGE NUM_PREPARED PROMPTS AND IT WILL GIVE REPEAT SAMPLES OF THE SAME PROMPT!

        for s, e in zip(start, end): #used to be while true but this is always going to be a high enough number. doesnt need to be an infinite loop!
            print('==================  start of this batch is:', s)
            print('we are at start index:', s, 'and end:', e)

            contexts_batch = []
            for ind in range(s, e):
                raw_text = df.loc[ind, 'Prompt And Completion']
                #print(' ========= raw text prompt ========== \n', raw_text)
                #print('========== end of raw text ==========')
                contexts_batch.append(enc.encode(raw_text))
            
            for c_len_ind, c in enumerate(contexts_batch): 
                if len(c) < to_perp_length:
                    print("prompt ", c_len_ind, 'is of insufficient length', len(c))


            '''shortest = to_perp_length #only want it to be 200 tokens. #len(min(contexts_batch)) # ensuring that the contexts are all the same length
            print('the max length all trimmed to is:', shortest)
            print('the shortest length is',len(min(contexts_batch)))
            print('the longest is:', len(max(contexts_batch)))
            contexts_batch = [c[:shortest] for c in contexts_batch]'''

            contexts_batch = [c[:to_perp_length] for c in contexts_batch if len(c)>=to_perp_length]
            
            generated = 0
            for _ in range(nsamples // batch_size): 
                out = sess.run(output, feed_dict={
                                    context: contexts_batch
                                })

                batch_perps = out[0]

                batch_logits = out[1]
                # rounding up their values. 
                #batch_perps = tf.cast(batch_perps, tf.float16)

                print(tf.shape(batch_perps))
                #adding to the list of all logits. 
                all_perplexities.append(batch_perps[:,prompt_length:])
                
                all_logits.append(batch_logits[:,prompt_length:,:])
                all_text.append(contexts_batch)
                #print('see what the first out looks like! before decoding', out[0])
                #print('out decoding index 0-50257', enc.decode(np.arange(0,50257)))

        #saving all of the logits into a pickle after all the prompts are iterated through:
        pickle.dump(all_perplexities, gzip.open(general_path+'gpt-2_output/'+'all_perplexities_'+experiment_name+'.pickle.gz', 'wb'))
        pickle.dump(all_logits, gzip.open(general_path+'gpt-2_output/'+'all_logits_'+experiment_name+'.pickle.gz', 'wb'))
        pickle.dump(all_text, gzip.open(general_path+'gpt-2_output/'+'all_text_'+experiment_name+'.pickle.gz', 'wb'))


if __name__ == '__main__':
    fire.Fire(interact_model)
