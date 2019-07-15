# iterate through the different conditions as outlined in the google docs

from interactive_conditional_samples import interact_model

vals_dict = {'tfs':[0.01, 0.05, 0.1, 0.5, 0.75 ], 'k':[200]  }

'''{'tfs':[None, 0.01, 0.05, 0.1, 0.5, 0.75 ],
'n': [0.1, 0.25, 0.5, 0.75, 0.9], 'k':[1,10,40,200]  }'''

variants_to_sample = ['tfs']#['n', 'k', 'tfs']

alpha_set = nuc_prob_set = top_k_set = 0

for samp_strat in variants_to_sample: 

    for val in vals_dict[samp_strat]: 

        if samp_strat=='tfs':
            alpha_set=val
        elif samp_strat=='n':
            nuc_prob_set=val
        else: 
            top_k_set=val

        interact_model( # some other variables are initialized below
            general_path = '',
            alpha=alpha_set,
            nuc_prob=nuc_prob_set,
            sampler=samp_strat, #n, k or tfs
            pre_prepared_prompts = True, 
            num_prepared_prompts_wanted = 100, #5000
            model_name='345M',
            seed=27,
            batch_size=25, # 500
            generated_length=150,
            prompt_length = 100,
            temperature=1,
            top_k=top_k_set,
            models_dir='../gpt-2/models',    
        )