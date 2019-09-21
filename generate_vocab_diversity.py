# iterate through the different conditions as outlined in the google docs
import random
from managing_vocab_diversity import interact_model

vals_dict = {'tfs':[0.25, 0.75, 0.9, 0.95, 0.99],
'n': [0.5, 0.63, 0.69, 0.81, 0.75, 0.9], 'k':[1,40,200]  }

#for seed in [28,56,10,40,9]: # in order to generate lots of examples for everything. 

alpha_set = flat_set = nuc_prob_set = top_k_set = 0

seed = 28

# randomly shuffling the order
keys =  list(vals_dict.keys())
random.shuffle(keys)
vals_dict = [(key, vals_dict[key]) for key in keys]
vals_dict = dict(vals_dict)

for samp_strat, values in vals_dict.items(): 

    for val in values: 

        if samp_strat=='tfs':
            alpha_set=val # this is actually now a probability threshold
        elif samp_strat=='n':
            nuc_prob_set=val
        elif samp_strat=='flat':
            flat_set=val
        else: 
            top_k_set=val

        interact_model( # some other variables are initialized below
            general_path = '',
            alpha=alpha_set,
            nuc_prob=nuc_prob_set,
            flat_prob = flat_set,
            sampler=samp_strat, #n, k or tfs
            pre_prepared_prompts = True, 
            num_prepared_prompts_wanted = 5000, #5000
            model_name='774M', # '345M',
            seed=seed,
            batch_size=25, # 500
            generated_length=150,
            prompt_length = 100,
            temperature=1,
            top_k=top_k_set,
            models_dir='../gpt-2/models',    
        )