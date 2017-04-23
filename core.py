import tensorflow as tf
import numpy as np


class TuringMachine(object):
	def __init__(self,memory_size,memory_shape,r_head,w_head,sess):
		with tf.variable_scope('TM'):
			# extract r_head variables
			self.r_k = r_head[0]
			self.r_b = r_head[1]
			self.r_g = r_head[2]
			self.r_s = r_head[3]
			self.r_t = r_head[4]

			self.w_k = w_head[0]
			self.w_b = w_head[1]
			self.w_g = w_head[2]
			self.w_s = w_head[3]
			self.w_t = w_head[4]

			self.N = memory_size
			self.M = memory_shape
		
			self.memory = tf.Variable(np.zeros((self.N,self.M)),tf.float32)
			
			self.r_w = tf.Variable(np.zeros((self.N)),tf.float32)
			self.w_w = tf.Variable(np.zeros((self.N)),tf.float32)
		self.sess = sess

		# w is one either r_w or w_w

	def get_weight_vector(self,k,b,g,s,t):
		mem_shape = tf.identity(self.memory).shape
		# prepare for content addressing
		# 1. 
		assert(tf.identity(k).shape == [self.M])
		assert(tf.identity(b).shape == [])

		ms = [tf.cast(tf.squeeze(m),tf.float32) for m in tf.split(self.memory,self.N)]
		ks = [tf.cast(k,tf.float32) for _ in range(self.N)]


		assert(len(ms) == self.N)
		km = [b * tf.reduce_sum(tf.multiply(k,m))/ (tf.norm(k) * tf.norm(m)) for (k,m) in zip(ks,ms)]
		
		# content addressing
		wc = tf.nn.softmax(km)
		
		# interpolation
		assert(tf.identity(g).shape == [])
		assert(wc.shape == [self.N])

		wg = g * wc + (1 - g) * tf.cast(self.r_w,tf.float32)
		# prepare for circular convolution
		assert(wg.shape == [self.N])
		with tf.control_dependencies([
			tf.assert_rank(s,1)
		]):
			wcon = [tf.Variable(0,tf.float32) for _ in range(self.N)]
			s_len = tf.identity(s).shape[0].__int__()
		# circular convolution
		assert(tf.identity(wcon).shape == [self.N]) # weight vector covers all memory grids 
		with tf.control_dependencies([
			tf.assert_rank(s,1),					# s is a vector
		]):
			# TODO: any faster?
			for i in range(self.N):
				for j in range(self.N):
					wcon[i] = wg[j] * s[(i - j) % s_len]

		wcon = tf.parallel_stack(wcon)
		# sharpening
		assert(tf.identity(wcon).shape == [self.N])	# weight vector still convers all memory grids
		assert(tf.identity(t).shape == [])				# t (gamma) is a scalar
		
		wt = wcon ** t
		return wt / tf.reduce_sum(wt)


	def _prepare_weight(self,w):
		return tf.transpose(tf.parallel_stack([w for _ in range(self.M)]))
	
	def get_read_weight(self):
		return self.get_weight_vector(self.r_k,self.r_b,self.r_g,self.r_s,self.r_t)

	def get_write_weight(self):
		return self.get_weight_vector(self.w_k,self.w_b,self.w_g,self.w_s,self.w_t)
	
	def read_op(self):
		# read_op = sum(wt * Mt)
		read_weight = tf.cast(self.get_read_weight(),tf.float64)
		read_op = tf.reduce_sum(tf.multiply(self._prepare_weight(read_weight),self.memory),0)
		update_op = tf.assign(self.r_w,read_weight)
		return read_op,update_op

	def write_op(self,erase_vec,add_vec):
		w_weights = tf.cast(self.get_read_weight(),tf.float64)
		assert w_weights.shape == [self.N]
		w = tf.cast(tf.expand_dims(w_weights,1),tf.float32)
		update_op = tf.assign(self.w_w,w_weights)
		e = 1 - tf.matmul(w,tf.expand_dims(erase_vec,1),transpose_b = True)
		a = tf.matmul(w, tf.expand_dims(add_vec,1),transpose_b = True)
		write_op = tf.assign(self.memory,tf.multiply(self.memory,tf.cast(e,tf.float64)) + tf.cast(a,tf.float64))
		return write_op,update_op

	

if __name__ == '__main__':
	# TM unit test
	with tf.Session() as sess:
		N = 10
		M = 2

		writer = tf.summary.FileWriter('tmp/core',sess.graph)
		erase_vec = tf.zeros([M])
		add_vec = tf.zeros([M])
		# k,b,g,s,t
		r_w = [tf.zeros([M]),tf.random_normal([]),tf.random_normal([]),tf.random_normal([N/2]),tf.random_normal([])]
		r_w = [tf.zeros([M]),tf.random_normal([]),tf.random_normal([]),tf.random_normal([N/2]),tf.random_normal([])]
		tm = TuringMachine(N,M,r_w,r_w,sess)
		sess.run(tf.global_variables_initializer())
		print tm.read_op(),tm.write_op(erase_vec,add_vec)

# abstract class of controllers. Which should expose the input,output tensors, parameters for each head as well as erase/add vector variables
class CopyNTM(object):
	def __init__(self,sess,input_dim,delim,mem_dim,num_memory,max_shift):
		self.sess = sess
		self.delim = delim
		# TODO: remove left hand side
		self.input,self.read_input,self.output,self.r_head,self.h_head,self.e_vec,self.a_vec = self.build_controller(input_dim,mem_dim,max_shift)
		self.tm = TuringMachine(num_memory,mem_dim,)
		
	def build_controller(self,input_dim,mem_dim,max_shift):
		assert len(input_dim) >= 2
		assert len(mem_dim) == 2

		self.input = tf.placeholder(input_dim,tf.float32)
		self.read_input = tf.placeholder(mem_dim,tf.float32)
		h = tf.concat(self.input,self.read_input,axis = 0)
		h = tf.layers.dense(h,200,tf.nn.relu)
		h = tf.layers.dense(h,100,tf.nn.relu)
		
		r = tf.layers.dense(h,50,tf.nn.relu)
		w = tf.layers.dense(h,50,tf.nn.relu)
		# k,b,g,s,t
		r_k = tf.layers.dense(r,mem_dim[0])
		r_b = tf.layers.dense(r,1,tf.nn.relu)
		r_g = tf.layers.dense(r,1,tf.nn.sigmoid)
		r_s = tf.layers.dense(r,max_shift)
		r_s = r_s / tf.norm(r_s) # normalize weight
		r_t = tf.layers.dense(r,1,tf.nn.relu)
		r_t = r_t + 1

		self.r_head = [r_k,r_b,r_g,r_s,r_t]

		w_k = tf.layers.dense(w,mem_dim[0])
		w_b = tf.layers.dense(w,1,tf.nn.relu)
		w_g = tf.layers.dense(w,1,tf.nn.sigmoid)
		w_s = tf.layers.dense(w,max_shift)
		w_s = w_s / tf.norm(w_s) # normalize weight
		w_t = tf.layers.dense(w,1,tf.nn.relu)
		w_t = w_t + 1
		
		self.e_vec = tf.layers.dense(w,mem_dim[0],tf.nn.relu)
		self.a_vec = tf.layers.dense(w,mem_dim[0])
		self.w_head = [w_k,w_b,w_g,w_s,w_t]


	
	def predict(self,input):
		pass
	
	def train(self,input,target):
		pass

