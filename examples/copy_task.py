import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from utils import *
from ops import *
from ntm import NTM

def generate_copy_data(batch_size,min_length,max_length,data_dim):
	inp_arr = np.zeros((max_length * 2 + 1,batch_size))
	out_arr = np.zeros((max_length * 2 + 1,batch_size))
	mask = np.zeros((max_length * 2 + 1,batch_size,data_dim))
	for b in range(batch_size):
		# generate the sequence length for the batch
		data_length = np.random.randint(min_length,max_length)
		inp_pattern = np.random.randint(1,data_dim - 1,(data_length,))
		# insert input
		inp_arr[:data_length,b] = inp_pattern
		inp_arr[data_length,b] = data_dim - 1
		out_arr[data_length + 1:2 * data_length + 1,b] = inp_pattern
		out_arr[data_length,b] = data_dim - 1

		mask[data_length + 1:2 * data_length + 1,b,:] = 1
	return inp_arr,out_arr,mask

def generate_sort_data(batch_size,min_length,max_length,data_dim):
	inp_arr = np.zeros((max_length * 2 + 1,batch_size))
	out_arr = np.zeros((max_length * 2 + 1,batch_size))
	mask = np.zeros((max_length * 2 + 1,batch_size,data_dim))
	for b in range(batch_size):
		data_length = np.random.randint(min_length,max_length)
		inp_pattern = np.random.randint(1,data_dim - 1,(data_length,))
		sorted_pattern = np.sort(inp_pattern,0)

		inp_arr[:data_length,b] = inp_pattern
		inp_arr[data_length,b] = data_dim - 1
		out_arr[data_length + 1:2 * data_length + 1,b] = sorted_pattern
		out_arr[data_length,b] = data_dim - 1
		mask[data_length + 1:2 * data_length + 1,b,:] = 1
	return inp_arr,out_arr,mask
if __name__ == '__main__':
	with tf.Session() as sess:
		batch_size = 4
		input_dim = 8
		output_dim = input_dim
		seq_len = 31
		mem_dim = output_dim
		controller_size = 20
		num_memory = 128
		num_read = 1
		num_write = 2
		org_input = tf.placeholder(tf.uint8,(seq_len,batch_size))
		org_target = tf.placeholder(tf.uint8,(seq_len,batch_size))

		inp = tf.one_hot(org_input,input_dim,axis = -1,dtype = tf.float32)
		target = tf.one_hot(org_target,input_dim,axis = -1,dtype = tf.float32)

		ntm = NTM('feed_forward',input_dim,output_dim,num_read,num_write,num_memory,mem_dim,controller_size,3,batch_size,scope = None)
		run_var = ntm.construct_run_var(inp)
		ntm_out = run_var[0]
		fin_memory = run_var[2]

		#mask_ph = tf.placeholder(tf.float32,(seq_len,batch_size,input_dim))
		# write on memory
		outputs = tf.unstack(ntm_out,seq_len,0)
		targets = tf.unstack(tf.cast(org_target,tf.int32),seq_len,0)
		loss = tf.reduce_mean([tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output,labels = t) for output,t in zip(outputs,targets)])
		

		opt = tf.train.RMSPropOptimizer(0.0001,momentum = 0.9,decay = 1e-6)
		gav = opt.compute_gradients(loss,tf.global_variables())
		gav = [(tf.clip_by_value(g,-10,10),v) for (g,v) in gav if g is not None]
		grad_summary = [tf.Print(g,[g],v.name + '::' + str(g.get_shape().as_list())) for (g,v) in gav]
		grads,vars = zip(*gav)
		train_op = opt.apply_gradients(gav)
		# copy task
		sess.run(tf.global_variables_initializer())

		# logging stuff
		writer = tf.summary.FileWriter('tmp/ntm',graph = sess.graph)
		saver = tf.train.Saver()
		inp_sum_ph = tf.expand_dims(tf.transpose(inp,perm = [1,2,0]),-1)
		target_sum_ph = tf.expand_dims(tf.transpose(target,perm = [1,2,0]),-1)
		output_sum_ph = tf.expand_dims(tf.transpose(ntm_out,perm = [1,2,0]),-1)
		mem_sum_ph = tf.expand_dims(tf.transpose(fin_memory,perm = [0,2,1]),-1)
		# summaries
		tf.summary.scalar('loss',loss)
		tf.summary.image('input',inp_sum_ph)
		tf.summary.image('target',target_sum_ph)
		tf.summary.image('output',output_sum_ph)
		tf.summary.image('memory',mem_sum_ph)
		for g in grads:
			shape = g.get_shape().as_list()
			if len(shape) == 2:
				tf.summary.image(g.name,tf.expand_dims(tf.expand_dims(g,-1),0))
		sums = tf.summary.merge_all()
		for epoch in range(400000):
			inp_pattern,oup_pattern,mask = generate_sort_data(batch_size,1,(seq_len - 1) / 2,input_dim)
			#print sess.run(ntm_out,feed_dict = {org_input: inp_pattern})
			loss_val,_,summaries = sess.run([loss,train_op,sums],
				feed_dict = {
					org_input: inp_pattern,
					org_target: oup_pattern,
					})
			writer.add_summary(summaries)
			print 'epoch: %d, loss %f' % (epoch,loss_val)

