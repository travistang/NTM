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

	
M = tf.Variable(np.array([[[1,1,1],[2,2,2],[1,1,1],[1,2,1]],[[1,1,1],[2,2,2],[1,1,1],[1,2,1]]]),dtype = tf.float32) # 1,4,3: batch_size,time step,mem_length
#M = tf.transpose(M,perm = [0,2,1]) # 1,3,4
wc = tf.Variable([[1,1,1,1],[1,1,1,1]],dtype = tf.float32)
k = tf.Variable([[1,1,1],[1,1,1]],dtype = tf.float32) # 1,3
b = tf.Variable([[1],[1]],dtype = tf.float32) # 1,1
g = tf.Variable([[0],[0]],dtype = tf.float32) #1,1
s = tf.Variable([[0,1,0],[0,1,0]],dtype = tf.float32) # 1,3
t = tf.Variable([[1],[1]],dtype = tf.float32)
e = tf.Variable([[1.0,1.0,1.0],[1.0,1.0,1.0]])
a = tf.Variable([[1.0,1.0,0.0],[0.0,0.0,0.0]])
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#new_w = read(k,b,g,s,t,M,wc)
	new_M = write(k,b,g,s,t,M,wc,e,a)
	print sess.run(new_M)[1]
	
