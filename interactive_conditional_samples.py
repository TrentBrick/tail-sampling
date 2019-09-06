#!/usr/bin/env python3
import fire
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import gzip
import model, sample, encoder
import time

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
    flat_prob = None,
    sampler='tfs', #n, k or tfs
    perc_acc=0.99,
    pre_prepared_prompts = True, 
    num_prepared_prompts_wanted = 100, #5000
    model_name='345M', #'774M',
    seed=27,
    batch_size=25, # 500
    generated_length=150,
    prompt_length = 100,
    softmax_output_size = 50527,
    temperature=1,
    top_k=0,
    models_dir='../gpt-2/models',    
):
    
    # initializing some other variables ==========
    nsamples=batch_size # should equal the batch size. 
    pre_prepared_prompts_data_path = general_path+'test_dataframe_500primer_only.csv'


    k_window_size =None
    window_weights = None
    if sampler=='tfs': #ADD IN OPTION TO NOT USE THE WEIGHTING AND COMPUTE FOR THE WHOLE THING
        sampling_param=alpha

        if sampling_param != None: 

            k_window_size = int(np.log(1-perc_acc)/np.log(1-alpha))

            if k_window_size>softmax_output_size: 
                k_window_size = softmax_output_size
        
            window_weights = (1-alpha)**np.arange(0,k_window_size) 
            window_weights = np.expand_dims(np.expand_dims(window_weights, axis=1), axis=2).astype(np.float32)
            print('size of K (window size) for EMA', k_window_size)

        else:
            print('Using an alpha of None')

    elif sampler=='n':
        sampling_param=nuc_prob

    elif sampler=='flat':
        sampling_param=flat_prob
        
    else: 
        sampling_param=top_k

    print('Using the sampling method:::', sampler, 'With parameter:::', sampling_param)

    experiment_name = "%s-sampling-type_%s-sampling-param_%s-word-prompts_%s-gen-length_%s-number-of-prompts_%s-seed_%s_model-parallelized" %(sampler,sampling_param,prompt_length, generated_length, num_prepared_prompts_wanted, seed, model_name)

    start = np.arange(0,num_prepared_prompts_wanted,batch_size)
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

    if generated_length is None:
        generated_length = hparams.n_ctx // 2
    elif generated_length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)


    #getting the random prompts that we want to use
    if pre_prepared_prompts==True: 
        np.random.seed(seed)
        print('dataframe being loaded in',pre_prepared_prompts_data_path)
        df=pd.read_csv(pre_prepared_prompts_data_path)
        rand_selections = np.random.randint(0,df.shape[0], size=num_prepared_prompts_wanted)       

    #saving all of the logits from the different batches taken in:
    all_logits = []

    #saving the actual text that was produced
    all_text = []

    batch_times = []

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=generated_length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, sampler=sampler, 
            top_k=top_k, alpha=alpha, nuc_prob=nuc_prob, flat_prob=flat_prob,
            k_window_size = k_window_size, window_weights=window_weights
    ) # 'n' is nucleus, 'k' is topk, 'tfs', is tail free sampling

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # CHANGE THIS TO BE IN RANGE NUM_PREPARED PROMPTS AND IT WILL GIVE REPEAT SAMPLES OF THE SAME PROMPT!
        for s, e in zip(start, end): #used to be while true but this is always going to be a high enough number. doesnt need to be an infinite loop!
            print('==================  start of this batch is:', s)
            print('we are at start index:', s, 'and end:', e)
            if pre_prepared_prompts==True:
                #generated further up ahead. 
                contexts_batch = []
                for ind in range(s, e):
                    raw_text = df.loc[rand_selections[ind], 'Prompt']
                    #print(' ========= raw text prompt ========== \n', raw_text)
                    #print('========== end of raw text ==========')
                    contexts_batch.append(enc.encode(raw_text))
                shortest = prompt_length #only want it to be 200 tokens. #len(min(contexts_batch)) # ensuring that the contexts are all the same length
                print('the max length all prompts are trimmed to is:', shortest)
                print('the shortest length is',len(min(contexts_batch)))
                print('the longest is:', len(max(contexts_batch)))
                contexts_batch = [c[:shortest] for c in contexts_batch]

            else: 
                raw_text = input("Model prompt >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Model prompt >>> ")
                context_tokens = enc.encode(raw_text)
                shortest = len(context_tokens)
            generated = 0
            for _ in range(nsamples // batch_size): # I have made this so that it is 1.
                start_timer = time.time()
                if pre_prepared_prompts==True: #making it so that I can feed multiple prompts into the batch!
                    out = sess.run(output, feed_dict={
                                        context: contexts_batch
                                    })

                else:
                    out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })

                end = time.time() - start_timer
                batch_times.append(end)

                batch_logits = out[1]

                # rounding up their values. 
                #batch_logits = tf.cast(batch_logits, tf.float16)

                out = out[0] # the original output which is the generated sequence
                
                all_text.append(out) # adding text including what the prompt was.
                
                out = out[:, shortest:] #gets rid of the prompt from the outputs. 
                
                print(tf.shape(batch_logits))
                #adding to the list of all logits. 
                all_logits.append(batch_logits)

                #print('see what the first out looks like! before decoding', out[0])
                #print('out decoding index 0-50257', enc.decode(np.arange(0,50257)))
                
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " GENERATED SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

        
        #end = end.numpy()
        print(' +++++++++++++++++++++ time taken to run sampling loop', end, '+++++++++++++++++++++++')
        #pickle.dump(batch_times, gzip.open(general_path+'gpt-2_output/'+'time_taken_for_all_'+experiment_name+'.pickle.gz', 'wb'))

        #saving all of the logits into a pickle after all the prompts are iterated through:
        pickle.dump(rand_selections, gzip.open(general_path+'gpt-2_output/'+'prompt_rand_selections_'+experiment_name+'.pickle.gz', 'wb'))
        pickle.dump(all_logits, gzip.open(general_path+'gpt-2_output/'+'all_logits_'+experiment_name+'.pickle.gz', 'wb'))
        pickle.dump(all_text, gzip.open(general_path+'gpt-2_output/'+'all_text_'+experiment_name+'.pickle.gz', 'wb'))

        

if __name__ == '__main__':
    fire.Fire(interact_model)
