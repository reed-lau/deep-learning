import tensorflow as tf
from load_mnist import *
#from network    import *
import numpy as np
import pprint


# Train/Test Data Input
train_data =load_mnist("mnist_fashion/train-images.idx3-ubyte")/256.0
train_label=load_mnist("mnist_fashion/train-labels.idx1-ubyte")
test_data  =load_mnist("mnist_fashion/t10k-images.idx3-ubyte")/256.0
test_label =load_mnist("mnist_fashion/t10k-labels.idx1-ubyte")

#train_data =load_mnist("mnist_data/train-images.idx3-ubyte")/256.0
#train_label=load_mnist("mnist_data/train-labels.idx1-ubyte")
#test_data  =load_mnist("mnist_data/t10k-images.idx3-ubyte")/256.0
#test_label =load_mnist("mnist_data/t10k-labels.idx1-ubyte")

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

image=tf.reshape(xs,(-1,28,28,1))

conv1=tf.layers.conv2d(image,32,3,1,'same',activation=tf.nn.relu)
pool1=tf.layers.max_pooling2d(conv1,2,2)

conv2=tf.layers.conv2d(pool1,32,5,1,'same',activation=tf.nn.relu)
pool2=tf.layers.max_pooling2d(conv2,2,2)

conv3=tf.layers.conv2d(pool2,64,3,1,'same',activation=tf.nn.relu)
pool3=tf.layers.max_pooling2d(conv3,2,2)

flat=tf.reshape(conv1,[-1,28*28*32])

yp=tf.layers.dense(flat,10,activation=tf.nn.softmax)


# Loss Function
#loss=tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=yp)
loss=-tf.reduce_mean( tf.reduce_sum( ys*tf.log(yp+1.0E-9),reduction_indices=[1] ) )

#train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train=tf.train.AdamOptimizer(0.001 ).minimize(loss)

accuracy=tf.metrics.accuracy(labels=tf.argmax(ys,axis=1),predictions=tf.argmax(yp,axis=1))[1]*100.0

init1=tf.global_variables_initializer()
init2=tf.local_variables_initializer()

sess=tf.Session()

sess.run(init1)
sess.run(init2)

print("Global Variables: liuwei")
pprint.pprint(tf.global_variables())

print("local Variables: liuwei")
pprint.pprint(tf.local_variables())

for ie in range(20000):
	i1=(ie%600)*100
	i2=i1+100
	sess.run(train,feed_dict={xs:train_data[i1:i2,:],ys:train_label[i1:i2,:]})
	
	if (ie%500==0):
		loss_,acc_=sess.run([loss,accuracy],feed_dict={xs:train_data,ys:train_label})
		acct=sess.run(accuracy,feed_dict={xs:test_data,ys:test_label})
		
		
		print('|step: %6.6d'%ie,'|loss: %6.5f '%loss_,'|acc: %3.1f%%'%acc_,'|acct: %3.1f%%'%acct)
		#print("step:",ie,"loss:",cost,"acc:",acc,"acct:",acct)