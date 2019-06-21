# iterate through the different conditions as outlined in the google docs

for : 

    for : 

        interact_model( # some other variables are initialized below
    general_path = '../../tail-sampling/',
    experiment_name = "5000_word_prompts",
    alpha=0.05,
    nuc_prob=0.25,
    sampler='tfs', #n, k or tfs
    pre_prepared_prompts = True, 
    num_prepared_prompts_wanted = 5, #5000
    model_name='345M',
    seed=27,
    batch_size=5, # 500
    length=100,
    temperature=1,
    top_k=1,
    models_dir='../models',    
)