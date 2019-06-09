import tensorflow as tf
import numpy as np

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!' # this is where the whole context is already given into the model. 
        # it is the primer that I write for it! 
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token) # this is not used in my case! 

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
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
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1),
                tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2) if all_logits is None else tf.concat([all_logits, tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2)], axis=2)
                #tf.concat([all_logits, tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2)], axis=2)
                #all_logits[:,:,tf.shape(output)[1]+1].assign(tf.expand_dims(next_outputs['logits'][:, -1, :], axis=2) )
            ]

        past, prev, output, all_logits = body(None, context, context, None) # for the first run the output and previous are both the context. 

        def cond(*args):
            return True

        _, _, tokens, all_logits_out = tf.while_loop(
                       cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output,
                all_logits
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, 50257, None]) #batch size
            ],
            back_prop=False,
        )

        return (tokens, all_logits_out)