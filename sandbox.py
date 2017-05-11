import tensorflow as tf
import numpy as np 

def debug_shape(t,name = None):
	print '%s has shape %s' % (name or t.name,str(t.get_shape().as_list()))
def debug_type(t,name):	
	print '%s has type %s' % (name or t.name,t.dtype)
def assert_rank(t,rank):
	assert rank == len(t.get_shape().as_list())
"""
	Input:
		weight vector of shape (b,w)
		matrix of shape (b,w,k)
	Output:
		Linear combination of shape (b,1) = sum(w_i * M_i)
"""
def linear_combination(w,M):
	assert_rank(w,2) # batch_size and weight vector
	assert_rank(M,3) # batch_size and 2D matrices (weights,keys)
	w = tf.expand_dims(w,1) # batch_size,_,weights
	x = tf.matmul(w,M) # batch_size,1,num_keys
	x = tf.squeeze(x,1)# batch_size,num_keys
	#x = tf.reduce_sum(x,0) # batch_size,1
	return x
"""
	Input:
		key vector of shape (b,k),
		matrix of shape (b,r,k)
	Output:
		cosine similarity of each column of the matrix and the key
		shape: (b,r)
"""
def cosine_similarity(k,M,eps = 1e-12):
	assert_rank(M,3) # batch_size and 2D matrics (r,v), where r is number of cells and v is the content of each cell
	assert_rank(k,2) # batch_size and key
	k = tf.expand_dims(k,1) # batch_size,_,key
	Mt = tf.transpose(M,perm = [0,2,1]) # batch_size,(v,r)
	sim = tf.matmul(k,Mt) # (batch_size x 1 x r)
	sim = tf.squeeze(sim,1) # (batch_size,r)
	
	norm_k = tf.norm(k)
	norm_M = tf.norm(M,axis = 2)
	denorm = norm_k * norm_M # batch_size = (b,r,1)
	
	return sim / (denorm + eps)
	
"""
	Input:
		key strength of shape (b,1)
		key of shape (b,k)
		memory of shape (b,n,k)
	Output:
		content address of shape (b,n)
"""	
def content_addressing(b,k,M):
	assert_rank(b,2) # batch_size and key_strength
	assert_rank(k,2) # batch_size and key_length
	assert_rank(M,3) # batch_size, num_memories and key_length
	cos_sim = cosine_similarity(k,M) # (b,r)
	b = tf.expand_dims(b,-1) 	 # (b,1,1)
	cos_sim = tf.expand_dims(cos_sim,1) # (b,1,r)
	addr = tf.matmul(b,cos_sim) # (b,1,r)
	addr = tf.squeeze(addr,1) # (b,r)
	return tf.nn.softmax(addr) # (b,r)
"""
	Input:
		Interpolation gate g of shape (b,1)
		current weight vector  wc of shape (b,r)
		previous weight vector wt_1 of shape (b,r)
	Output:
		interpolated weight of shape (b,r)
"""
def interpolate(g,wc,wt_1):
	assert_rank(g,2)	# (b,1)
	assert_rank(wc,2) 	# (b,r)
	assert_rank(wt_1,2)	# (b,r)
	g = tf.expand_dims(g,-1)
	wc = tf.expand_dims(wc,1)
	wt_1 = tf.expand_dims(wt_1,1)
	res = tf.matmul(g,wc) + tf.matmul(1 - g,wt_1) # (b,1,r)
	return tf.squeeze(res,1) # (b,r)	
"""
	Input:
		Shift vector s of shape (b,s)
		weight vector w of shape (b,r)
	Output:
		shifted weight of shape (b,r)
"""
def shift_location(s,w):
	assert_rank(s,2)
	assert_rank(w,2)
	pad_length = tf.shape(s)[-1] / 2
	w_shape = tf.shape(w)
	batch_size = w_shape[0]
	weight_length = w_shape[-1]
	w = tf.concat([tf.slice(w,[0,weight_length - pad_length],[batch_size,pad_length]),w,tf.slice(w,[0,0],[batch_size,pad_length])],-1)	
	""" prepare for conv 2d"""
	debug_shape(w,'w')
	debug_shape(s,'s')
	w = tf.expand_dims(w,1) # b,1,r
	w = tf.expand_dims(w,-1) # b, 1, r ,1: "Batch",h,w,in_channel
	s = tf.expand_dims(s,-1) # b,s,1
	s = tf.expand_dims(s,-1) # b,s,1,1: batch, w,h,out_channel
	batch_size = w.get_shape().as_list()[0]	
	ws = tf.split(w,batch_size,0)
	ss = tf.split(s,batch_size,0)
	"""conv"""
	res = [tf.nn.conv2d(w,s,[1 for _ in range(4)],'VALID') for (w,s) in zip(ws,ss)]	
	res = [tf.squeeze(tf.squeeze(r,0),-1) for r in res]
	res = tf.stack(res,0)
	res = tf.squeeze(res,1)
	return res
"""
	Input:
		sharpening factor g of shape (b,1)
		weight vector w of shape (b,r)
	Output:
		sharpened weight vector of shape (b,r)
"""
def sharpening(g,w):
	assert_rank(g,2)
	assert_rank(w,2)
	key_length = tf.shape(w)[-1]
	g = tf.tile(g,[1,key_length])
	res = tf.pow(w,g)
	return tf.nn.l2_normalize(res,-1)
	#return res / tf.reduce_sum(res,-1)

def get_weight(k,b,g,s,t,M,wt):
	wc = content_addressing(b,k,M)	
	wg = interpolate(g,wc,wt)
	ws = shift_location(s,wg)	
	w_fin = sharpening(t,ws)
	return w_fin
	#return linear_combination(w_fin,M)

def read(k,b,g,s,t,M,wt):
	w = get_weight(k,b,g,s,t,M,wt)
	return linear_combination(w,M)

def write(k,b,g,s,t,M,wt,e,a):
	e = tf.clip_by_value(e,0,1) # values of e should be restricted to [0,1]
	assert_rank(e,2) # e should have shape (b,k)
	assert_rank(a,2) # a should have shape (b,k)
	w = get_weight(k,b,g,s,t,M,wt) # w has shape (b,n)
	w = tf.expand_dims(w,-1) # (b,n,1)
	""" erase """
	e = tf.expand_dims(e,1) # (b,1,k)
	e = 1 - tf.matmul(w,e) # (b,n,k)
	""" add """
	a = tf.expand_dims(a,1) # (b,1,k)
	a = tf.matmul(w,a) # (b,n,k)
	return tf.multiply(M,e) + a # (b,n,k) as new contents of the memory

class NTM(object):
	def __init__(self,controller_type,inp_ph,out_dim,num_read,num_write,num_memory,mem_length,controller_size,shit_range,scope = None):
		self.input_var = inp_ph
		self.input_dim = tf.shape(self.input_var)[-1]
		self.read_vars = []
		self.write_vars = []
		self.out_dim = out_dim
		self.num_read = num_read
		self.num_write = num_write
		self.num_memory = num_memory
		self.mem_length = mem_length
		self.shift_range = shift_range
		self.controller_dim = controller_size
		
		# construct graph for read heads
		read_op = self.build_head_op(read = True)
		write_op = self.build_head_op(read = False)
		""" construct internal graph """
		"""
			Structure:
		"""
		with tf.variable_scope(self.scope or 'ntm'):
			if controller_type = "feed_forward":
				# TODO weight
			else:
				raise NotImplementedError('Unsupported controller type: %s' % controller_type)	
	def main_loop(self,old_vars,x_t):
		# extract all variables
		num_read_head_params = len(list('rgbstw'))
		num_write_head_params= len(list('rgbstwea')) 
		sep_ind = self.num_read * num_read_head_params
		read_vars = old_vars[:sep_ind]
		write_vars = old_vars[sep_ind:self.num_write * num_write_head_params]
		

		# map all the read variables to the read op
		# TODO: what about memories?
		read_vecs = [read(*read_vars[i:i + num_read_head_params]) for i in range(0,len(read_vars),num_read_head_params)]
		write_vecs = [write(*write_vars[i:i + num_write_head_params]) for i in range(0,len(write_vars),num_write_head_params)]
		
		# collect read head params for next loop
		"""
			Architecture for feedforward controller:
				input = concat(x_t,rv1,rv2....)
				con1 = Dense(input,controller_size,'relu')
				con2 = Dense(con1,controller_size,'relu')
				out = Dense(con2,out_dim,'relu')
		"""	
		with tf.variable_scope(self.scope or 'ntm'):
			inp = tf.concat([x_t] + read_vecs,-1)
			con1 = tf.nn.relu(tf.nn.xw_plus_b(inp,self.W_con1,self.b_con1)
			con2 = tf.nn.relu(tf.nn.xw_plus_b(con1,self.W_con2,self.b_con2)
			out = tf.nn.softmax(tf.nn.sw_plus_b(con2,self.W_out,self.b_out)
			# TODO: dynamic read head graph construction
			# TODO: put the weight for the controller to the def of the class
		# TODO: what about write head?	
	def get_read_vars(self,id = 0):
		num_var = len(list('rbgstw'))
		res = self.read_vars[num_var * id: num_var * id + num_var] # k,b,g,s,t,wt
		assert res != []
		return res
	def get_write_vars(self,id = 0):
		num_var = len(list('rbgstwea'))
		res = self.write_vars[num_var * id: num_var * id + num_var] # k,b,g,s,t,wt,e,a
		assert res != []
		return res
	def build_haed_op(self,M,read = True):
		if read:
			r = tf.Variable(np.zeros(None,self.mem_length))
			b = tf.Variable(np.zeros(None,1))
			g = tf.Variable(np.zeros(None,1))
			s = tf.Variable(np.zeros(None,self.shift_range))
			t = tf.Variable(np.zeros(None,1))
			wt = tf.Variable(np.zeros(None,self.num_memory))
			self.read_vars.append(r)
			self.read_vars.append(b)
			self.read_vars.append(g)
			self.read_vars.append(s)
			self.read_vars.append(t)
			self.read_vars.append(wt)
			return read(r,b,g,s,t,M,wt)
		else:
			r = tf.Variable(np.zeros(None,self.mem_length))
			b = tf.Variable(np.zeros(None,1))
			g = tf.Variable(np.zeros(None,1))
			s = tf.Variable(np.zeros(None,self.shift_range))
			t = tf.Variable(np.zeros(None,1))
			wt = tf.Variable(np.zeros(None,self.num_memory))
			e = tf.Variable(np.zeros(None,self.mem_length))
			a = tf.Variable(np.zeros(None,self.mem_length))
			self.write_vars.append(r)
			self.write_vars.append(b)
			self.write_vars.append(g)
			self.write_vars.append(s)
			self.write_vars.append(t)
			self.write_vars.append(wt)
			self.write_vars.append(e)
			self.write_vars.append(a)	
			return write(r,b,g,s,t,M,wt,e,a)

M = tf.Variable(np.array([[[1,1,1],[2,2,2],[1,1,1],[1,2,1]],[[1,1,1],[2,2,2],[1,1,1],[1,2,1]]]),dtype = tf.float32) # 1,4,3: batch_size,time step,mem_length
#M = tf.transpose(M,perm = [0,2,1]) # 1,3,4
wc = tf.Variable([[1,1,1,1],[1,1,1,1]],dtype = tf.float32)
k = tf.Variable([[1,1,1],[1,1,1]],dtype = tf.float32) # 1,3
b = tf.Variable([[1],[1]],dtype = tf.float32) # 1,1
g = tf.Variable([[0],[0]],dtype = tf.float32) #1,1
s = tf.Variable([[0,1,0],[0,1,0]],dtype = tf.float32) # 1,3
t = tf.Variable([[1],[1]],dtype = tf.float32)
e = tf.Variable([[1.0,1.0,1.0],[1.0,1.0,1.0]])
a = tf.Variable([[1.0,1.0,1.0],[0.0,0.0,0.0]])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#new_w = read(k,b,g,s,t,M,wc)
	new_M = write(k,b,g,s,t,M,wc,e,a)
	print sess.run(new_M)[1]
	
