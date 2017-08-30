import numpy as np
import struct 

def load_mnist(filename):
	fp= open(filename,"rb")
	hd=fp.read(4)
	
	magic=struct.unpack('>i',hd[:4])[0]
	
	# 2049-train-label
	# 2051-train-image
	# 2049-test-label
	# 2051-test-image
	if (magic==2049 ):
		hd=fp.read(4)
		nsmp=struct.unpack('>i',hd[0:4])[0]
		nrow=1
		ncol=1
	elif (magic==2051):
		hd=fp.read(12)
		nsmp=struct.unpack('>i',hd[0:4])[0]
		nrow=struct.unpack('>i',hd[4:8])[0]
		ncol=struct.unpack('>i',hd[8:12])[0]
			
	bdata=fp.read()
	
	fp.close()
	
	nb=len(bdata)
	if (nb!=nsmp*nrow*ncol):
		print("bdata=",nb,"file length error")
		return
	
	if (magic==2049):
		a=np.zeros((nsmp,ncol*nrow*10),dtype=np.float32)
		for i in range(nsmp):
			a[i,ord(bdata[i])]=1.0	
	elif (magic==2051):
		a=np.zeros((nsmp,ncol*nrow),dtype=np.float32)
		for i in range(nsmp):
			for ix in range(nrow*ncol):
				a[i,ix]=ord(bdata[i*nrow*ncol+ix])	
	else:
		a=None
	
	return a


if __name__ == '__main__' :
	import matplotlib.pyplot as plt
	
	train_data =load_mnist("mnist_data/train-images.idx3-ubyte")
	train_label=load_mnist("mnist_data/train-labels.idx1-ubyte")
	test_data  =load_mnist("mnist_data/t10k-images.idx3-ubyte")
	test_label =load_mnist("mnist_data/t10k-labels.idx1-ubyte")
	
	print(train_data.shape )
	print(train_label.shape)
	print(test_data.shape  )
	print(test_label.shape )
	
	plt.ion()
	while(1):
		i=np.random.randint(0,train_data.shape[0])
		data=train_data[i].reshape((28,28))
		title=('%d' % np.argmax(train_label[i,:]))

		plt.imshow(data)
		plt.title(title)
		plt.show()
		plt.pause(0.5)
		plt.cla()
		