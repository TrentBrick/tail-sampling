import tensorflow as tf
import numpy as np
import model

def ema_calc(vals, alpha):
    inv_alpha = 1-alpha
    for i in np.arange(vals.shape[1].value):
        #print('value of i in the ema calc: ', i)
        if i == 0:
            emas = tf.expand_dims(vals[:,i], axis=1)
        else: 
            emas = tf.concat([emas, tf.expand_dims(tf.math.multiply(vals[:,i],alpha)+tf.math.multiply(inv_alpha,emas[:,i-1]), axis=1)], axis=1)
    return tf.reverse(emas, axis=[1])

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def ema_eff(alpha, vals, k_window_size, window_weights ):

    padding = k_window_size -1
    # THIS CAN BE DONE IN A BATCH V EFFICIENTLY
    #out = torch.nn.functional.conv1d(torch.from_numpy(vals).unsqueeze(0).unsqueeze(1).double(),torch.from_numpy(window_weights).unsqueeze(0).unsqueeze(1), padding=p )
    
    print('ema eff shape', vals.shape)

    padding = k_window_size -1

    tensor_paddings = tf.constant([[0, 0], 
                            [padding, padding],
                            [0,0]])
    print('vals shape pre conv op', vals.shape)

    # need to give the output vals a channel: 
    vals = tf.expand_dims(vals, axis=2 )

    vals = tf.pad(vals, tensor_paddings, "CONSTANT")
    out = tf.cast(tf.nn.conv1d(vals, window_weights, padding='VALID', stride=1 ), tf.float32) #.astype(tf.float32)
    print('out shape', out.shape)
    out = tf.math.multiply(alpha,out[:,padding:,0])
    print('out shape', out.shape)
    return out

def tail_free(logits, alpha, k_window_size, window_weights):
    #print('logits passed into the tfs', logits.shape)
    soft = tf.nn.softmax(logits, axis=1)
    indices = tf.argsort(logits, direction='DESCENDING', axis=1)
    if alpha is not None:
        sps = ema_eff(alpha, soft, k_window_size, window_weights)
        '''#print('sps passed into the ema', logits.shape)
        sps = tf.sort(soft, direction='ASCENDING',axis=1)
        sps = ema_calc(sps, alpha)'''
    else: 
        sps = tf.sort(soft, direction='DESCENDING',axis=1) #isnt doing any of the reversing
    sps = my_tf_round(sps, 2) # quantization
    grad = sps[:,1:]-sps[:,:-1] # first derivative
    grad = grad[:,1:]-grad[:,:-1] #this is the 2nd derivative

    tail_ids = tf.cast(grad.shape[1].value- tf.argmax( tf.cast(tf.greater(tf.reverse(grad,axis=[1]), 0.001),tf.int8) ,axis=1 ), tf.int32)
    
    while_condition = lambda i, logits_to_return: tf.less(i, logits.shape[0].value)
    
    def body(i, logits_to_return):
        #print('RUNNING THE TFS BODY CODE!!! ')
        ids_above_tail = indices[i,:tail_ids[i]+1]
        logit_mask = tf.sparse_to_dense( tf.sort(ids_above_tail, direction='ASCENDING',axis=0), [logits.shape[1].value,], 0.0, 1.0)*-1e10
        logit = logits[i, :] + logit_mask
        return [tf.add(i, 1),
               tf.expand_dims(logit, axis=0) if logits_to_return is None else tf.concat([logits_to_return, tf.expand_dims(logit, axis=0)], axis=0) 
        ]
    
    i = tf.constant(0, dtype=tf.int32)
    _, logits_to_return = body(i, None)
    i = tf.constant(1, dtype=tf.int32)
    _, logits_to_return = tf.while_loop(while_condition, body, [i, logits_to_return], shape_invariants=[i.get_shape(), 
                                      tf.TensorShape([None, logits.shape[1].value])] )
    
    '''
    tail_id_mask = tf.sequence_mask(tail_ids, logits.shape[1].value)
    ids_above_tails = tf.where(tail_id_mask, indices, tf.ones_like(tail_id_mask, dtype=tf.int32)*-10)
    logits_mask = tf.cast(tf.math.logical_not(tf.sequence_mask(ids_above_tails, logits.shape[1].value)), tf.float32)*-1e10 # now selecting only the bad logits.    
    logits = logits+logits_mask'''

    return logits_to_return #returning the selected logits. the multinomial takes in logits not softmax. 

def nucleus(logits, p):
    indices = tf.argsort(logits, direction='DESCENDING', axis=1)
    vals = tf.sort(tf.nn.softmax(logits, axis=1), direction='DESCENDING',axis=1)
    tail_ids = tf.cast(tf.argmax(tf.cast(tf.cumsum(vals, axis=1)>p, tf.int8), axis=1), tf.int32)

    while_condition = lambda i, logits_to_return: tf.less(i, logits.shape[0].value)
    def body(i, logits_to_return):
        ids_above_tail = indices[i,:tail_ids[i]+1]
        logit_mask = tf.sparse_to_dense( tf.sort(ids_above_tail, direction='ASCENDING',axis=0), [logits.shape[1].value,], 0.0, 1.0)*-1e10
        logit = logits[i, :] + logit_mask
        return [tf.add(i, 1),
               tf.expand_dims(logit, axis=0) if logits_to_return is None else tf.concat([logits_to_return, tf.expand_dims(logit, axis=0)], axis=0) 
        ]
    
    i = tf.constant(0, dtype=tf.int32)
    _, logits_to_return = body(i, None)
    i = tf.constant(1, dtype=tf.int32)
    _, logits_to_return = tf.while_loop(while_condition, body, [i, logits_to_return], shape_invariants=[i.get_shape(), 
                                      tf.TensorShape([None, logits.shape[1].value])] )
    
    return logits_to_return


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


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, 
sampler='k', temperature=1, top_k=0, alpha=0.05, nuc_prob=0.25, 
k_window_size=None, window_weights=None):
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

        return (tokens, all_logits_out)