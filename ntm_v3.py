from utils import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import math

def generate_copy_sequence(batch_size,data_size,time_step,sess):
	called_count = 0
	seq_input = tf.placeholder(tf.int64,(batch_size,time_step))
	#sequence = np.random.randint(0,data_size - 1,(batch_size,time_step))
	#prepare the three main areas of input and output: zeros, pattern and stop symbols
	seq = tf.map_fn(lambda n: tf.one_hot(n,data_size,dtype = tf.int64), seq_input)
	zeros = tf.zeros([batch_size,time_step,data_size],tf.int64)
	stop = tf.map_fn(lambda n: tf.one_hot(n,data_size,dtype = tf.int64),(data_size - 1) * np.ones((batch_size,1),dtype = np.int64))
        long_stop = tf.map_fn(lambda n: tf.one_hot(n,data_size,dtype = tf.int64),(data_size - 1) * np.ones((batch_size,time_step),dtype = np.int64))

	# then concat them
	inp = tf.concat([seq,stop,long_stop],1)
	out = tf.concat([zeros,stop,seq],1)
	#mask = tf.cast(tf.reduce_max(out,2),tf.float32)
	while True:
			yield sess.run([tf.cast(inp,tf.float32),tf.cast(out,tf.float32)],feed_dict = {seq_input: np.random.randint(0,data_size - 1,(batch_size,time_step))})

class NTMCell(tf.contrib.rnn.RNNCell):
	def __init__(self,sess,input_shape,output_shape,batch_size = 16,memory_shape = (128,40),controller_size = 200,shift_range = 3,scope = None):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.memory_shape = memory_shape
		self.num_memory = memory_shape[0]
		self.key_length = memory_shape[-1]
		self.shift_range = shift_range
		self.batch_size = batch_size
		self.controller_size = controller_size

		self.scope = scope

		self.sess = sess
                
                self.max_time_ph = tf.placeholder(tf.int32)
		self.M_0 = tf.Variable(np.random.rand(*memory_shape)) * 1e-3
		self.r_t0 = tf.zeros((self.key_length,))
		self.rw_0 = tf.zeros((self.num_memory,))
		self.ww_0 = tf.zeros((self.num_memory,))


                # summary stuff
                self.summaries = []
		with tf.variable_scope(self.scope or 'rnn'):
			with tf.variable_scope('Weights'):
				# controller weights
				# input-read -> hidden
				W_con = tf.get_variable('W_con',shape = (self.input_shape + self.key_length,controller_size),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_con = tf.get_variable('b_con',shape = (controller_size,),initializer = tf.constant_initializer(0))
				# hidden -> out
				W_out = tf.get_variable('W_out',shape = (controller_size,self.output_shape),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_out = tf.get_variable('b_out',shape = (self.output_shape,),initializer = tf.constant_initializer(0))
				
				# read head weights
				W_rk = tf.get_variable('W_rk',shape = (controller_size,self.key_length),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_rk = tf.get_variable('b_rk',shape = (self.key_length,),initializer = tf.constant_initializer(0))
				
				W_rb = tf.get_variable('W_rb',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_rb = tf.get_variable('b_rb',shape = (1,),initializer = tf.constant_initializer(0))

				W_rg = tf.get_variable('W_rg',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_rg = tf.get_variable('b_rg',shape = (1,),initializer = tf.constant_initializer(0))

				W_rs = tf.get_variable('W_rs',shape = (controller_size,self.shift_range),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_rs = tf.get_variable('b_rs',shape = (self.shift_range,),initializer = tf.constant_initializer(1e-3))

				W_rt = tf.get_variable('W_rt',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_rt = tf.get_variable('b_rt',shape = (1,),initializer = tf.constant_initializer(0))

				# write head weights
				W_wk = tf.get_variable('W_wk',shape = (controller_size,self.key_length),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_wk = tf.get_variable('b_wk',shape = (self.key_length,),initializer = tf.constant_initializer(0))
				
				W_wb = tf.get_variable('W_wb',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_wb = tf.get_variable('b_wb',shape = (1,),initializer = tf.constant_initializer(0))

				W_wg = tf.get_variable('W_wg',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_wg = tf.get_variable('b_wg',shape = (1,),initializer = tf.constant_initializer(0))

				W_ws = tf.get_variable('W_ws',shape = (controller_size,self.shift_range),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_ws = tf.get_variable('b_ws',shape = (self.shift_range,),initializer = tf.constant_initializer(1e-3))

				W_wt = tf.get_variable('W_wt',shape = (controller_size,1),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_wt = tf.get_variable('b_wt',shape = (1,),initializer = tf.constant_initializer(0))
				# erase/add vectors
				W_e  = tf.get_variable('W_e',shape = (self.controller_size,self.key_length),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_e  = tf.get_variable('b_e',shape = (self.key_length,),initializer = tf.constant_initializer(0))
				
				W_a  = tf.get_variable('W_a',shape = (self.controller_size,self.key_length),initializer = tf.random_normal_initializer(stddev = 1e-3))
				b_a  = tf.get_variable('b_a',shape = (self.key_length,),initializer = tf.constant_initializer(1e-3))
                
                # write weights
                self.trainable_weights = [W_a,b_a,W_e,b_e,W_wt,b_wt,W_ws,b_ws,W_wg,b_wg,W_wb,b_wb,W_wk,b_wk]
                # read weights
                self.trainable_weights += [W_rt,b_rt,W_rs,b_rs,W_rg,b_rg,W_rb,b_rb,W_rk,b_rk]

	def get_traininable_weights(self):
		return self.trainable_weights

	def cosine_similarity(self,M, k, eps=1e-6):
		k = tf.expand_dims(k,1)
		Mt = tf.transpose(M,perm = [0,2,1])
		kt = tf.transpose(k,perm = [0,2,1])
		norm_k = tf.reduce_sum(tf.matmul(k,kt))
		norm_M = tf.reduce_sum(tf.matmul(M,Mt))
		dot = tf.matmul(k,Mt)
		return dot / (norm_k * norm_M + eps)

	# thanks to https://github.com/carpedm20/NTM-tensorflow/blob/76588d73a00fdbc3f2b19b9c95e35de3c0a4f2b0/ops.py
	def circular_convolution(self,v, k):
		"""Computes circular convolution.
		Args:
		    v: a 1-D `Tensor` (vector)
		    k: a 1-D `Tensor` (kernel)
		"""

		size = int(v.get_shape()[0])
		kernel_size = int(k.get_shape()[0])
		kernel_shift = int(math.floor(kernel_size/2.0))

		def loop(idx):
			if idx < 0: return size + idx
			if idx >= size : return idx - size
			else: return idx

		kernels = []
		for i in xrange(size):
			indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
			v_ = tf.gather(v, indices)

			kernels.append(tf.reduce_sum(v_ * k, 0))
		return tf.dynamic_stitch([i for i in xrange(size)], kernels)
 	
 	def batch_circular_convolution(self,M,k):
 		batch_size = M.get_shape().as_list()[0]
 		vks = zip(tf.split(tf.squeeze(M,1),batch_size,0),tf.split(k,batch_size,0))
 		res = [self.circular_convolution(tf.squeeze(v,0),tf.squeeze(k,0)) for v,k in vks]
 		#return tf.map_fn(lambda vk: circular_convolution(vk[0],vk[1]),vks)
 		return tf.stack(res,0)

 	# input: described in paper
 	# wt: the last read weight
	def eval_out(self,k,b,g,s,t,M,wt):
		assert k.get_shape().as_list() == [self.batch_size,self.key_length]
		assert b.get_shape().as_list() == [self.batch_size,1]
		assert g.get_shape().as_list() == [self.batch_size,1]
		assert s.get_shape().as_list() == [self.batch_size,self.shift_range]
		assert t.get_shape().as_list() == [self.batch_size,1]
		#assert M.get_shape().as_list() == [self.batch_size] + list(self.memory_shape)
		#assert wt.get_shape().as_list() == [self.batch_size,self.num_memory]
		# content addressing
		wc = tf.nn.softmax(tf.exp(tf.expand_dims(b,-1) * self.cosine_similarity(M,k)))
		# interpolation
		lg = tf.expand_dims(g,-1)
		wt = tf.expand_dims(wt,1)
		wg = tf.matmul(lg,wc) + tf.matmul(1 - lg,wt)
		# circulat convolution ( weight shifting)
		wcon = self.batch_circular_convolution(wg,s)
		# sharpening
		pow_t = tf.concat([t for _ in range(100)],1)
		wf = tf.nn.softmax(tf.pow(wcon,pow_t))
		return wf

	@property
	def state_size(self):
            return (self.M_0.get_shape(),self.r_t0.get_shape(),self.rw_0.get_shape(),self.ww_0.get_shape())

	@property
	def output_size(self):
            return self.output_shape


	def __call__(self,inputs,state,scope = None):
            return self.step(state,inputs,scope)
	# the function to be used to scan throught the list of input
	# x_t is the input vector of A TIME STEP with shape (batch_size x input_size)
	def step (self,(M_t,r_t,rw_t,ww_t),(x_t),scope = None):
                fprint('step...')
                # log tensors
		with tf.variable_scope('Weights',reuse = True):
			# controller weights
			# input-read -> hidden
			W_con = tf.get_variable('W_con',shape = (self.input_shape + self.key_length,self.controller_size))
			b_con = tf.get_variable('b_con',shape = (self.controller_size,))
			# hidden -> out
			W_out = tf.get_variable('W_out',shape = (self.controller_size,self.output_shape))
			b_out = tf.get_variable('b_out',shape = (self.output_shape,))
			# read head weights
			W_rk = tf.get_variable('W_rk',shape = (self.controller_size,self.key_length))
			b_rk = tf.get_variable('b_rk',shape = (self.key_length,))
			W_rb = tf.get_variable('W_rb',shape = (self.controller_size,1))
			b_rb = tf.get_variable('b_rb',shape = (1,))

			W_rg = tf.get_variable('W_rg',shape = (self.controller_size,1))
			b_rg = tf.get_variable('b_rg',shape = (1,))

			W_rs = tf.get_variable('W_rs',shape = (self.controller_size,self.shift_range))
			b_rs = tf.get_variable('b_rs',shape = (self.shift_range,))

			W_rt = tf.get_variable('W_rt',shape = (self.controller_size,1))
			b_rt = tf.get_variable('b_rt',shape = (1,))

			# write head weights
			W_wk = tf.get_variable('W_wk',shape = (self.controller_size,self.key_length))
			b_wk = tf.get_variable('b_wk',shape = (self.key_length,))
			
			W_wb = tf.get_variable('W_wb',shape = (self.controller_size,1))
			b_wb = tf.get_variable('b_wb',shape = (1,))

			W_wg = tf.get_variable('W_wg',shape = (self.controller_size,1))
			b_wg = tf.get_variable('b_wg',shape = (1,))

			W_ws = tf.get_variable('W_ws',shape = (self.controller_size,self.shift_range))
			b_ws = tf.get_variable('b_ws',shape = (self.shift_range,))

			W_wt = tf.get_variable('W_wt',shape = (self.controller_size,1))
			b_wt = tf.get_variable('b_wt',shape = (1,))

			W_e  = tf.get_variable('W_e',shape = (self.controller_size,self.key_length))
			b_e  = tf.get_variable('b_e',shape = (self.key_length,))
				
			W_a  = tf.get_variable('W_a',shape = (self.controller_size,self.key_length))
			b_a  = tf.get_variable('b_a',shape = (self.key_length,))

		# construct graph for NTM
		with tf.variable_scope(scope or type(self).__name__):
                        debug_shape(x_t,'x_t')
			#x_t = tf.squeeze(x_t,1)
			long_inp = tf.concat([x_t,r_t],-1)
			con_hidden = tf.nn.relu(tf.nn.xw_plus_b(long_inp,W_con,b_con))
			con_out = tf.nn.relu(tf.nn.xw_plus_b(con_hidden,W_out,b_out))
			
			rk = tf.nn.xw_plus_b(con_hidden,W_rk,b_rk)
			rb = tf.nn.relu(tf.nn.xw_plus_b(con_hidden,W_rb,b_rb))
			rg = tf.nn.sigmoid(tf.nn.xw_plus_b(con_hidden,W_rg,b_rg))
			rs = tf.nn.softmax(tf.nn.xw_plus_b(con_hidden,W_rs,b_rs))
			rt = 1 + tf.nn.relu(tf.nn.xw_plus_b(con_hidden,W_rt,b_rt))

			wk = tf.nn.xw_plus_b(con_hidden,W_wk,b_wk)
			wb = tf.nn.relu(tf.nn.xw_plus_b(con_hidden,W_wb,b_wb))
			wg = tf.nn.sigmoid(tf.nn.xw_plus_b(con_hidden,W_wg,b_wg))
			ws = tf.nn.softmax(tf.nn.xw_plus_b(con_hidden,W_ws,b_ws))
			wt = 1 + tf.nn.relu(tf.nn.xw_plus_b(con_hidden,W_wt,b_wt))

			ev = tf.nn.xw_plus_b(con_hidden,W_e,b_e)
			av = tf.nn.xw_plus_b(con_hidden,W_a,b_a)

			# eval read/write weight
			r_weight = self.eval_out(rk,rb,rg,rs,rt,M_t,rw_t)
			w_weight = self.eval_out(wk,wb,wg,ws,wt,M_t,ww_t)

			assert r_weight.get_shape().as_list() == w_weight.get_shape().as_list() == [self.batch_size,self.num_memory]
			
			# read vector
			# input: read_weight: (batch_size,num_mem)
			# input: mem_shape: (batch_size,num_mem,key_length)
			# output shape: (batch_size x key_length)
			r_t1 = tf.squeeze(tf.matmul(tf.expand_dims(r_weight,1),M_t),1)

			# weight vector
			# input: write_weight: ( batch_size,num_mem)
			# input: erase_vector: ( batch_size, key_length)
			# input: add_vector: (batch_size,key_length)
			# intermediate output: elementwise erase matrix (batch_size,num_mem,key_length)
			assert ev.get_shape().as_list() == av.get_shape().as_list() == [self.batch_size,self.key_length]
			em = tf.matmul(tf.expand_dims(w_weight,-1),tf.expand_dims(ev,1))
			am = tf.matmul(tf.expand_dims(w_weight,-1),tf.expand_dims(av,1))
			M_t1 = tf.multiply(M_t,1 - em) + am

		# handle step-to-step stuff

		return (con_out,(M_t1,r_t1,r_weight,w_weight))



if __name__ == '__main__':
	with tf.Session() as sess:
		seq_length = 10
		org_seq_length = 2 * seq_length + 1
		input_shape = 8
		target_shape = 8
		batch_size = 1
		clip_vals = 10
		lr = 0.0001

                mask = make_masks(batch_size,org_seq_length)
                inp_var = tf.placeholder(tf.float32,(batch_size,org_seq_length,input_shape))
                inp_var_time_major = tf.transpose(inp_var,perm = [1,0,2])
		inputs = tf.split(inp_var,org_seq_length,1)

		target_var = tf.placeholder(tf.float32,(batch_size,input_shape))
		gen = generate_copy_sequence(batch_size,input_shape,seq_length,sess)

                tm =  NTMCell(sess,input_shape,target_shape,memory_shape = (100,3),batch_size = batch_size,controller_size = 20)
		output, (M_fin,r_fin,rw_fin,ww_fin) = tf.nn.dynamic_rnn(tm,inp_var_time_major,dtype= tf.float32,time_major = True)

                output_time_minor = tf.transpose(output,perm = [1,0,2])

		org_target = tf.placeholder(tf.float32,(batch_size,org_seq_length,target_shape))
		target  = tf.argmax(org_target,2)
		weights = tf.placeholder(tf.float32,(batch_size,org_seq_length))

		loss_op = tf.contrib.seq2seq.sequence_loss(output_time_minor,target,weights)
                opt = tf.train.RMSPropOptimizer(lr)
                print 'comuting gradients...'
		grads = opt.compute_gradients(loss_op,tm.get_traininable_weights())
                print 'unused gradients: %s' % str([(g,v) for (g,v) in grads if g is None])
		grads = [(tf.clip_by_value(g,-clip_vals,clip_vals),v) for (g,v) in grads if g is not None]
		train_op = opt.apply_gradients(grads)
                """summaries"""
                read_summary = tf.summary.histogram('read_vector',r_fin)
                rw_summary = tf.summary.histogram('read_weights', rw_fin)
                ww_summary = tf.summary.histogram('write_weights', ww_fin)
                input_summary = tf.summary.image('input',tf.expand_dims(inp_var,-1))
                output_summary = tf.summary.image('output',tf.expand_dims(output_time_minor,-1))
                target_summary = tf.summary.image('target',tf.expand_dims(org_target,-1))
                loss_summary = tf.summary.scalar('loss',loss_op)
                M_summary = tf.summary.image('memory',tf.expand_dims(M_fin,-1))
                weight_summaries = log_dense_weights(tm.get_traininable_weights())
                merge_summary = tf.summary.merge([loss_summary,M_summary,input_summary,output_summary,target_summary,read_summary,rw_summary,ww_summary] + weight_summaries)
                """end of summaries"""
                writer = tf.summary.FileWriter('tmp/ntm_v3',graph = sess.graph)
                sess.run(tf.global_variables_initializer())

                for epoch in range(70000):
                    inp,oup = gen.next()
                    feed_dict = {
                                org_target: oup,
                                weights: mask,
                                inp_var: inp,
                            }
                    loss,summary,_ = sess.run([loss_op,merge_summary,train_op],feed_dict = feed_dict)
                    writer.add_summary(summary,epoch)
                    print 'epoch:{},loss:{}'.format(epoch,loss)
