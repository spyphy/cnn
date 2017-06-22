import visual
import sys
import network4
from network4 import Network
from network4 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network4.load_data_shared()

mini_batch_size = 10

#net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

net = Network([	SoftmaxLayer(n_in=784, n_out=10)], mini_batch_size)

#net.SGD(outfile, training_data, validation_data, test_data, mini_batch_size, epoch, eta ) 
# net.SGD(training_data, validation_data, test_data, mini_batch_size, epochs=1, eta=0.2, k_eta=0.99998, teaching=True)  


[shared_image_data, image_data] = network4.load_image_data()

#net.CALC(test_data, mini_batch_size)  
res_arr = net.SGD(shared_image_data, shared_image_data, shared_image_data, mini_batch_size, epochs=20, eta=0.2, k_eta=0.99998, teaching=False)  

print(res_arr)

y = image_data[1]
print(y[0:10])

x = image_data[0]
num = 8
digit = x[num]
visual.show_image(digit)

import cPickle as pickle
import gzip
dump = pickle.dumps(net)
f = gzip.open('net_dump.gz', 'wb')
f.write(dump)
f.close()

