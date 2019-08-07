import tensorflow as tf
import numpy as np
import model

"""
Calculates the perplexities at every time point for the input sentence. 
"""

def log2(x):
    # first correct for any 0 values. 
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def perp_calc(*, hparams, length, start_token=None, batch_size=None, context=None):
    
    lm_output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)

    print('logits pre seelction of 0',lm_output['logits'].shape)

    probs = tf.nn.softmax(lm_output['logits'], axis=2)

    # this has all of the logits for the entire context. 
    batch_perplexities = tf.math.pow(2.0, ( - tf.reduce_sum( probs*log2(probs+0.000000001), axis=2)))
    batch_logits = lm_output['logits']

    return (batch_perplexities, batch_logits )

def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!' # this is where the whole context is already given into the model. 
        # it is the primer that I write for it! 
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token) # this is not used in my case! 

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        #print('results from the model', lm_output.shape)
        #print('logits pre seelction of 0',lm_output['logits'].shape)

        logits = lm_output['logits'][:, :, :hparams.n_vocab] # only does anything if trying to restrict the vocab length. 
        #print('shape of the lgotis', logits.shape)
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):

        # this will store all of the logits for a specific sampling run. 
        # ultimately I want this to be a:  samples * prompts (batch size) * words * length  sized matrix. 

        #all_logits = tf.Variable(tf.zeros([batch_size, 50257, length]), dtype=tf.float32) 
        #all_logits = tf.Variable(name='all_logits', shape=[batch_size, 50257, length], initializer= ,dtype=tf.float32, trainable=False) 

        def body(past, prev, output, all_logits):
            next_outputs = step(hparams, prev, past=past)

            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if sampler=='k':
                print('using top k')
                logits = top_k_logits(logits, k=top_k)
            elif sampler=='n':
                print('using nucleus')
                logits = nucleus(logits, p=nuc_prob)
            elif sampler=='tfs':
                print('using tail free sampling')
                logits = tail_free(logits, alpha, k_window_size, window_weights)
            else: 
                print('defauling to top k sampling')
                logits = top_k_logits(logits, k=top_k)
            #print('the logits shape post processing is: ', logits.shape)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            #print('the samples shape is: ', samples.shape)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                tf.reshape(samples,[batch_size,1]),
                tf.concat([output, samples], axis=1),
                tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2) if all_logits is None else tf.concat([all_logits, tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2)], axis=2)
                #tf.concat([all_logits, tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2)], axis=2)
                #all_logits[:,:,tf.shape(output)[1]+1].assign(tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2) )
            ]

        past, prev, output, all_logits = body(None, context, context, None) # for the first run the output and previous are both the context. 

        def cond(*args):
            return True

        start = tf.timestamp()

        _, _, tokens, all_logits_out = tf.while_loop(
                       cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output,
                all_logits
            ],
            #changed the 2nd shape invariant so that it can handle the ? shape (which is actually batch size) for the TFS sampling. 
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, 50257, None]) #batch size
            ],
            back_prop=False,
        )

        end = tf.timestamp() - start
        print(' +++++++++++++++++++++ time taken to run sampling loop', end, '+++++++++++++++++++++++')

        return (tokens, all_logits_out )