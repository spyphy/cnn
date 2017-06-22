import visual
import sys
from PIL import Image
import numpy as np
import theano
import theano.tensor as T

import network4
#from network4 import Network

mini_batch_size = 10

import cPickle as pickle
import gzip
f = gzip.open('net_dump.gz', 'rb')
net = pickle.load(f)
f.close()

training_data, validation_data, test_data = network4.load_data_shared()

[shared_image_data, image_data] = network4.load_image_data()

#----------

# load an image from file
image_file_name = 'a.png'
im = Image.open(image_file_name)
if im.size != 28:
	print('Resize. The old size is ' + str(im.size))
	im = im.resize((28, 28))	

def convert_to_linear_array(img):
	mas = []
	nx, ny = img.size
	p = img.load()
	for y in range(0, ny):
		for x in range(0, nx):
			val = sum(p[x,y])/(256.0*3.0)
			mas.append(val)	
	return np.array(mas, dtype=np.float32)

def shared(data): # This allows Theano to copy the data to the GPU.
	shared_x = theano.shared(
		np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
	shared_y = theano.shared(
		np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
	return shared_x, T.cast(shared_y, "int32")
	
im_lin = convert_to_linear_array(im)
print('len im_lin = ' + str(len(im_lin)))
list_x = [im_lin for i in range(0,10)]
list_y = [0 for i in range(0,10)]
data_x = np.array(list_x, dtype=np.float32)
data_y = np.array(list_y, dtype=np.int64)
image_data = (data_x, data_y)
shared_image_data = shared(image_data)

#----------

y = image_data[1]
res_goal = y[0:mini_batch_size]

#training_data = shared_image_data
#validation_data = shared_image_data
res_out = net.SGD(training_data, validation_data, shared_image_data, mini_batch_size, epochs=1, eta=0.2, k_eta=0.99998, teaching=False)
print('out:  ' + str(res_out))
print('goal: ' + str(res_goal))
res =[1 if res_out[i]==res_goal[i] else 0 for i in range(0, mini_batch_size)]
print(res)

res_out = net.SGD(training_data, validation_data, shared_image_data, mini_batch_size, epochs=1, eta=0.2, k_eta=0.99998, teaching=False)
print('out:  ' + str(res_out))
print('goal: ' + str(res_goal))
res =[1 if res_out[i]==res_goal[i] else 0 for i in range(0, mini_batch_size)]
print(res)

x = image_data[0]
num = 8
digit = x[num]
visual.show_image(digit)





