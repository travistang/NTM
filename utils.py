
import numpy as np
import tensorflow as tf

def assert_rank(t,rank):
	assert rank == len(t.get_shape().as_list())
def fprint(s):
    print '{}\r'.format(s)
def debug_shape(t,name = None):
	print '%s has shape %s' % (name or t.name,t.get_shape().as_list())

def debug_type(t,name):	
	print '%s has type %s' % (name or t.name,t.dtype)

def log_dense_weights(ts):
    summaries = []
    for t in ts:
        rank = len(t.get_shape().as_list())
        if rank == 2: # W
            summaries.append(tf.summary.image(t.name,tf.expand_dims(tf.expand_dims(t,0),-1)))
        elif rank == 1: # b
            summaries.append(tf.summary.histogram(t.name,t))
        else:
            print 'unable to add summary for %s' % t.name
    return summaries

def prepare_image_summary(t,name):
    shape = t.get_shape().as_list()
    if len(shape) == 3:
        t = tf.expand_dims(t,0)
    if len(shape) == 2:
        t = tf.expand_dims(t,0)
        t = tf.expand_dims(t,-1)

    assert len(t.get_shape().as_list()) == 4
    return tf.summary.image(name,t)

def make_masks(batch_size,seq_length):
    res = np.zeros((batch_size,seq_length))
    data_len = (seq_length - 1) / 2
    res[:,data_len + 1:] = 1
    return res

