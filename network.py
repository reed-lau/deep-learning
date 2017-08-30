import tensorflow as tf

def addlayer(inputs,insize,outsize,actfun=None):
	W=tf.Variable(tf.random_normal([insize,outsize]))
	b=tf.Variable(tf.zeros([1,outsize])+0.1)

	xWpb=tf.matmul(inputs,W)+b

	if actfun is None:
		outputs=xWpb
	else:
		outputs=actfun(xWpb)
	return outputs

def addconv2d(input,Wshape=(3,3,1,32),actfun=None):
	W=tf.Variable(tf.truncated_normal(Wshape,stddev=0.1))
	b=tf.Variable(tf.zeros((Wshape[-1]))+0.1)
	
	conv=tf.nn.conv2d(input,W,[1,1,1,1],padding='SAME')+b

	if actfun is None:
		outputs=conv
	else:
		outputs=actfun(conv)
	
def addmaxpool(input,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME'):
	return tf.nn.max_pool(input,ksize=ksize,strides=strides,padding=padding)