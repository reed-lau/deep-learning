import tensorflow as tf
from load_mnist import *
from network    import *
import numpy as np


# Train/Test Data Input
train_data =load_mnist("mnist_fashion/train-images.idx3-ubyte")/256.0
train_label=load_mnist("mnist_fashion/train-labels.idx1-ubyte")
test_data  =load_mnist("mnist_fashion/t10k-images.idx3-ubyte")/256.0
test_label =load_mnist("mnist_fashion/t10k-labels.idx1-ubyte")

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

print(train_data.max())
print(train_label.max())


# Input PlaceHolder
xs=tf.placeholder(tf.float32,[None,train_data.shape[1]])
ys=tf.placeholder(tf.float32,[None,train_label.shape[1]])

# NetWork Structure
layer1=addlayer(xs,train_data.shape[1],20,actfun=tf.nn.sigmoid)
layer2=addlayer(layer1,20,20,actfun=tf.nn.sigmoid)
yp    =addlayer(layer2,20,train_label.shape[1],actfun=tf.nn.softmax)

# Loss Function
loss=-tf.reduce_mean(tf.reduce_sum(ys*tf.log(yp),reduction_indices=[1]))

train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

for ie in range(1000000):
	i1=(ie%600)*100
	i2=i1+100
	sess.run(train,feed_dict={xs:train_data[i1:i2,:],ys:train_label[i1:i2,:]})
	
	if (ie%500==0):
		cost=sess.run(loss,feed_dict={xs:train_data,ys:train_label})
		ys1=sess.run(yp,feed_dict={xs:train_data,ys:train_label})
		ys2=sess.run(yp,feed_dict={xs:test_data,ys:test_label})
		
		acc = np.sum(np.argmax(ys1,axis=1)==np.argmax(train_label,axis=1))/ys1.shape[0]*100.0	
		acct= np.sum(np.argmax(ys2,axis=1)==np.argmax(test_label,axis=1))/ys2.shape[0]*100.0	
		
		print("step:",ie,"loss:",cost,"acc:",acc,"acct:",acct)