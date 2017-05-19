
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

def to_one_hot(t,data_len):
	return tf.one_hot(t,data_len,axis = -1)

def to_label(t):
	return tf.argmax(t,axis = -1)

def generate_copy_data(batch_size,min_seq_length,max_seq_length,data_dim):
	inputs = []
	outputs = []
	for _ in range(batch_size):
		data_len = np.random.randint(min_seq_length,max_seq_length)
		seq = np.random.randint(1,data_dim,(data_len,))
		# add stop symbol
		inp = np.concatenate((seq,[0 for i in range(max_seq_length - data_len + 1)]))
		# replicate data
		out = np.concatenate(([0 for i in range(max_seq_length - data_len + 1)],seq))
		inputs.append(inp)
		outputs.append(out)
	# return (time_step,batch_size, 1)
	return np.stack(inputs,axis = 1),np.stack(outputs,axis = 1)
