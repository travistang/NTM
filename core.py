import tensorflow as tf
import numpy as np


class TuringMachine(object):
	def __init__(self,memory_size,memory_shape,r_head,w_head,sess):
		with tf.name_scope('TM'):
			self.r_w = tf.Variable(np.zeros(self.N))
			self.w_w = tf.Variable(np.zeros(self.N))

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
		
			self.memory = tf.Variable(np.zeros((self.N,self.M)))
		
		self.sess = sess

		# w is one either r_w or w_w

	def get_weight_vector(self,k,b,g,s,t):
		with tf.control_dependencies([tf.identity(k).shape == [self.M],tf.identity(b.shape) == []]):
			ms = [tf.squeeze(m) for m in tf.split(self.memory)]
			ks = [k for _ in range(self.N)]

		with tf.control_dependencies([len(ms) == self.N]):
			km = [b * tf.matmul(k,m)/ (tf.norm(k) * tf.norm(m)) for (k,m) in zip(ks,ms)]
		
		wc = tf.nn.softmax(km)

		with tf.control_dependencies([
			tf.identity(g).shape == [],
			wc.shape == tf.identity(self.memory).shape
		]):
			wg = g * wc + (1 - g) * self.memory
		with tf.control_dependencies([
			tf.assert_rank(s,1)
		]):
			
			for i in range(N - 1):
				wc[i] 


	def _prepare_weight(self,w):
		assert w in [self.r_w,self.w_w]
		return tf.transpose(tf.parallel_stack([w for _ in range(self.M)]))
	
	def read(self):
		# read_op = sum(wt * Mt)
		read_op = tf.reduce_sum(tf.multiply(self._prepare_weight(self.r_w),self.memory),0)
		return self.sess.run(read_op)

	def write(self,erase_vec,add_vec):
		pass

	

if __name__ == '__main__':
	with tf.Session() as sess:
		r_w = tf.Variable(np.random.rand(10))
		w_w = tf.Variable(np.random.rand(10))
		tm = TuringMachine(10,2,r_w,w_w,sess)
		sess.run(tf.global_variables_initializer())
		print tm.read()
