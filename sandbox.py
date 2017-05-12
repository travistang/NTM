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
	w_shape = w.get_shape().as_list()
	batch_size = w_shape[0]
	weight_length = w_shape[-1]
	w = tf.concat([tf.slice(w,[0,weight_length - pad_length],[batch_size,pad_length]),w,tf.slice(w,[0,0],[batch_size,pad_length])],-1)	
	""" prepare for conv 2d"""
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
	""" 
		enforce the shape of the resultant tensor 
		from (batch_size, ?) to (batch_size,memory_size) 
	"""
	res = tf.reshape(res,w_shape)
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

def write(w,M,e,a):
	e = tf.clip_by_value(e,0,1) # values of e should be restricted to [0,1]
	assert_rank(e,2) # e should have shape (b,k)
	assert_rank(a,2) # a should have shape (b,k)
	w = tf.expand_dims(w,-1) # (b,n,1)
	""" erase """
	e = tf.expand_dims(e,1) # (b,1,k)
	e = 1 - tf.matmul(w,e) # (b,n,k)
	""" add """
	a = tf.expand_dims(a,1) # (b,1,k)
	a = tf.matmul(w,a) # (b,n,k)
	res = tf.multiply(M,e) + a # (b,n,k) as new contents of the memory
	return res
class NTM(object):
	def __init__(self,controller_type,input_dim,out_dim,num_read,num_write,num_memory,mem_length,controller_size,shift_range,batch_size,scope = None):
		self.input_dim = input_dim 
		self.controller_vars = []
		self.read_vars = [] # contains the weight of the fc layers of the read head
		self.write_vars = [] # contains the weight of the fc layers of the write head
		self.out_dim = out_dim
		self.batch_size = batch_size
		self.num_read = num_read
		self.num_write = num_write
		self.num_memory = num_memory
		self.mem_length = mem_length
		self.shift_range = shift_range
		self.controller_dim = controller_size
		self.scope = scope	
		# construct graph for read heads
		""" construct internal graph """
		"""
			Structure:
		"""
		with tf.variable_scope(self.scope or 'ntm'):	
			num_read_head_params = len(list('rgbstw'))
			num_write_head_params= len(list('rgbstwea')) 

			if controller_type == "feed_forward":

				self.W_con1 = tf.Variable(np.random.rand(self.mem_length * self.num_read + self.input_dim,self.controller_dim),dtype=tf.float32)
				self.b_con1 = tf.Variable(np.zeros(self.controller_dim,),dtype=tf.float32)
				self.W_con2 = tf.Variable(np.random.rand(self.controller_dim,self.controller_dim),dtype=tf.float32)	
				self.b_con2 = tf.Variable(np.zeros(self.controller_dim,),dtype=tf.float32)
				self.W_out = tf.Variable(np.random.rand(self.controller_dim,self.out_dim),dtype=tf.float32)
				self.b_out = tf.Variable(np.zeros(self.out_dim,),dtype=tf.float32)
				# store all the variables above
				self.controller_vars = [self.W_con1,self.b_con1,self.W_con2,self.b_con2,self.W_out,self.b_out]
			else:
				raise NotImplementedError('Unsupported controller type: %s' % controller_type)	
		
			[self.build_head_params(self.controller_dim,read = True) for _ in range(self.num_read)]
			[self.build_head_params(self.controller_dim,read = False) for _ in range(self.num_write)]


	"""
		Return the collection of traininable variables of the NTM
	"""
	def get_trainable_params(self):
		return self.controller_vars + self.read_vars + self.write_vars

	""" Create params of the fully connected layer  """
	def build_head_params(self,input_dim,read = True):
		Wr = tf.Variable(np.random.rand(input_dim,self.mem_length),dtype = tf.float32)
		br = tf.Variable(np.zeros(self.mem_length,),dtype = tf.float32)
		Wb = tf.Variable(np.random.rand(input_dim,1),dtype = tf.float32)
		bb = tf.Variable(np.zeros(1,),dtype = tf.float32)
		Wg = tf.Variable(np.random.rand(input_dim,1),dtype = tf.float32)
		bg = tf.Variable(np.zeros(1,),dtype = tf.float32)
		Ws = tf.Variable(np.random.rand(input_dim,self.shift_range),dtype = tf.float32)
		bs = tf.Variable(np.zeros(self.shift_range,),dtype = tf.float32)
		Wt = tf.Variable(np.random.rand(input_dim,1),dtype = tf.float32)
		bt = tf.Variable(np.zeros(1,),dtype = tf.float32)
		if read:
			self.read_vars += [Wr,br,Wb,bb,Wg,bg,Ws,bs,Wt,bt]
		else: # write
			We = tf.Variable(np.random.rand(input_dim,self.mem_length),dtype = tf.float32)
			be = tf.Variable(np.zeros(self.mem_length,),dtype = tf.float32)
			Wa = tf.Variable(np.random.rand(input_dim,self.mem_length),dtype = tf.float32)
			ba = tf.Variable(np.zeros(self.mem_length,),dtype = tf.float32)
			self.write_vars += [Wr,br,Wb,bb,Wg,bg,Ws,bs,Wt,bt,We,be,Wa,ba]

	""" 
		get a tuple of tensors as the initial state of tf.scan function
		the tuple will be in the format of (M,(r1,r2,...rn),(rw1,rw2,...rwn),(ww1,ww2,...wwm)
		where num_read = n, num_write = m and ri is the read vector emitted by read head i, rwi,wwi are read/write weights evaluated by previous weights.
	"""
	def initial_state(self,inp_ph):
		seq_len = inp_ph.size()
		output = tf.TensorArray(tf.float32,seq_len,dynamic_size = False)
		count = tf.constant(0)
		M0 = tf.zeros((self.batch_size,self.num_memory,self.mem_length))
		rvs = tuple([tf.zeros((self.batch_size,self.mem_length),dtype=tf.float32) for _ in range(self.num_read)])
		rws = tuple([tf.zeros((self.batch_size,self.num_memory),dtype=tf.float32) for _ in range(self.num_read)])
		wws = tuple([tf.zeros((self.batch_size,self.num_memory),dtype=tf.float32) for _ in range(self.num_write)])
		return [output,count,M0,rvs,rws,wws,inp_ph]
		
		
	"""
		Main loop of NTM
		To be applied on tf.scan.
		At time step t, main_loop takes the following parameters:
			1. memory M_t
			2. tuple of read vectors rts
			3. tuple of read weights rw_ts 
			4. tuple of write weights ww_ts
			5. input x_t at time t of shape (batch_size,input_dim)
		And does the following:
			1. evaluate all head parameters (k,b,g,s,t,...) using controller weights with input x_t and all read vectors rts
			2. evaluate next read vectors and weights using read and write functions
			3. aggregate resultant read vectors and weights
			4. write memory M
			5. return new memory and all vectors, weights
	"""
	def main_loop(self,output,count,M_t,rts,rw_ts,ww_ts,in_tensor):
		x_t = in_tensor.read(count)
		# collect read head params for next loop
		"""
			Architecture for feedforward controller:
				input = concat(x_t,rv1,rv2....)
				con1 = Dense(input,controller_size,'relu')
				con2 = Dense(con1,controller_size,'relu')
				out = Dense(con2,out_dim,'relu')
		"""
		with tf.variable_scope(self.scope or 'ntm'):
			inp = tf.concat([x_t] + list(rts),-1)
			con1 = tf.nn.relu(tf.nn.xw_plus_b(inp,self.W_con1,self.b_con1))
			con2 = tf.nn.relu(tf.nn.xw_plus_b(con1,self.W_con2,self.b_con2))
			out = tf.nn.softmax(tf.nn.xw_plus_b(con2,self.W_out,self.b_out))
	
			new_read_weights = []
			new_read_vectors = []
			new_write_weights = []
			ea_tuples = []
			for _ in range(self.num_read):
				""" Construct read heads. The build_read_head function should return the read vector op and read weight op"""
				rv_op,rw_op = self.build_read_head(con2,M_t,rw_ts,_)
				new_read_vectors.append(rv_op)
				new_read_weights.append(rw_op)

			for _ in range(self.num_write):
				""" Construct write heads. The build_write_head function should return the write weight op """
				ww_op,ea = self.build_write_head(con2,M_t,ww_ts,_)
				new_write_weights.append(ww_op)
				ea_tuples.append(ea)
	
		# turn list to tuples
			new_read_weights = tuple(new_read_weights)
			new_read_vectors = tuple(new_read_vectors)	
			new_write_weights = tuple(new_write_weights)

		# write memory
		# apply write weight operations to the memory one by one
		# TODO: is this ok?!
			M = M_t
			for w,(e,a) in zip(new_write_weights,ea_tuples):
				M = write(w,M,e,a)

		# prepare for next state
			output = output.write(count,out)
		return [output,count + 1,M,new_read_vectors,new_read_weights,new_write_weights,in_tensor]
	"""
		Construct the operators from inp_ph tensor to read weight
		Input:
			inp_ph: The input tensor the subgraph starts with
			M:	The memory tensor
			rw_t:	The read weights, should be a list of tensorrs
			id:	The id of read weights, should be a scalar in range [0,num_read)
		Output:
			A tuple (rv,rw) that contains 2 tensors
			First one is the read vector op
			Second one is the read weight op
	"""
	def build_read_head(self,inp_ph,M,rw_t,id):
		# extract read vars ( The fc weights)
		num_vars = 10 # k,b,g,s,t, each has W and b
		vars = self.read_vars[id * num_vars: (id + 1) * num_vars]
		Wk,bk,Wb,bb,Wg,bg,Ws,bs,Wt,bt = vars
		wt = rw_t[id]	
		# construct subgraph
		k = tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wk,bk))
		b = tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wb,bb))
		g = tf.nn.sigmoid(tf.nn.xw_plus_b(inp_ph,Wg,bg))
		s = tf.nn.softmax(tf.nn.xw_plus_b(inp_ph,Ws,bs))
		t = 1 + tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wt,bt))
		
		# weight ops
		rw = get_weight(k,b,g,s,t,M,wt)
		# read ops
		rv = linear_combination(rw,M)	
		return (rv,rw)	
	"""
		Similar to build_read_head,except the output is a tuple of 2 tensors ( write weights,write_op)
	"""
	def build_write_head(self,inp_ph,M,ww_t,id):
		# extract read vars ( The fc weights)
		num_vars = 14 # k,b,g,s,t,e,a, each has W and b
		vars = self.write_vars[id * num_vars: (id + 1) * num_vars]	
		Wk,bk,Wb,bb,Wg,bg,Ws,bs,Wt,bt,We,be,Wa,ba = vars
		wt = ww_t[id]
		# construct subgraph
		k = tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wk,bk))
		b = tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wb,bb))
		g = tf.nn.sigmoid(tf.nn.xw_plus_b(inp_ph,Wg,bg))
		s = tf.nn.softmax(tf.nn.xw_plus_b(inp_ph,Ws,bs))
		t = 1 + tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wt,bt))
		e = tf.nn.sigmoid(tf.nn.xw_plus_b(inp_ph,We,be))
		a = tf.nn.relu(tf.nn.xw_plus_b(inp_ph,Wa,ba))
		
		# weight ops
		ww = get_weight(k,b,g,s,t,M,wt)
		
		return (ww,(e,a))
	"""
		construct tf.while_loop that consumes the input in ph
		The input placeholder will first be unstacked into a TensorArray,
		the resultant vectors will then be consumed in the while loop
		The output TensorArray will be stacked before return
		Input:
			the input placeholder of shape (seq_len,batch_size,input_dim)
		Output:
			a list of tensors:
				1. output tensor of shape (seq_len,batch_size,output_dim)
				2. The read vectors emitted by the read heads after consuming the last input
				3. The read weights
				4. The write weights
				5. The original input
	"""
	def construct_run_var(self,inp_ph):
		assert_rank(inp_ph,3)
		inp_shape = inp_ph.get_shape().as_list()
		assert inp_shape[1] == self.batch_size
		assert inp_shape[2] == self.input_dim
		seq_len = inp_shape[0]
		# decompose the input tensor and store into a TensorArray
		inp_ph = tf.TensorArray(tf.float32,seq_len,clear_after_read = False).unstack(inp_ph)
		# prepare for while loop
		loop_vars = self.initial_state(inp_ph)
		cond = lambda output,count,M,new_read_vectors,new_read_weights,new_write_weights,inp: tf.less(count,seq_len)
		res = tf.while_loop(cond,self.main_loop,loop_vars)
		# combine the output TensorArray to Tensor
		res[0] = res[0].stack()
		# combine the input TensorArray to Tensor
		res[-1] = res[-1].stack()
		return res	

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
	batch_size = 16
	input_dim = 40
	output_dim = 10
	seq_len = 100
	mem_dim = 20
	inp = tf.placeholder(tf.float32,(seq_len,batch_size,input_dim))
	ntm = NTM('feed_forward',input_dim,output_dim,1,1,128,mem_dim,60,3,batch_size,scope = None)
	run_var = ntm.construct_run_var(inp)
	sess.run(tf.global_variables_initializer())
	print sess.run(run_var[0],feed_dict = {inp: np.zeros((seq_len,batch_size,input_dim))})
