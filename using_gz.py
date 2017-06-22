import visual
import sys
import network4
#from network4 import Network

mini_batch_size = 10

import cPickle as pickle
import gzip
f = gzip.open('net_dump.gz', 'rb')
net = pickle.load(f)
f.close()

#training_data, validation_data, test_data = network4.load_data_shared()

[shared_image_data, image_data] = network4.load_image_data()

y = image_data[1]
res_goal = y[0:mini_batch_size]

training_data = shared_image_data
validation_data = shared_image_data
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
visual.save_image_png(digit)




