import sys
import network4
from network4 import Network
from network4 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network4.load_data_shared()
from network4 import ReLU

mini_batch_size = 10

#net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

net = Network([
	ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),  filter_shape=(20, 1, 5, 5),  poolsize=(2, 2)),
	ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2)),
	ConvPoolLayer(image_shape=(mini_batch_size, 40, 4, 4),   filter_shape=(80, 40, 3, 3), poolsize=(2, 2)),
	#FullyConnectedLayer(n_in=80*1*1, n_out=100),
	#FullyConnectedLayer(n_in=32*1*1, n_out=20, activation_fn=ReLU, p_dropout=0.5),
	#FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
	SoftmaxLayer(n_in=80, n_out=10)], mini_batch_size)

#net.SGD(outfile, training_data, validation_data, test_data, mini_batch_size, epoch, eta ) 
net.SGD(training_data, validation_data, test_data, mini_batch_size, epochs=20, eta=0.5, k_eta=0.99998)  

make_dump = True
if make_dump:
	import cPickle as pickle
	import gzip
	dump = pickle.dumps(net)
	f = gzip.open('net_dump.gz', 'wb')
	f.write(dump)
	f.close()
