import loadData
import tensorflow as tf
from tensorflow.python import debug as tfdbg

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name='weight_init')
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name='bias_init')
    return tf.Variable(initial,name=name)

def conv2d(x,W,name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',name=name)

bulider = loadData.olivettifaces_bulider()

sess = tf.InteractiveSession()

x  = tf.placeholder("float",shape=[None,57*47],name='x')
y_ = tf.placeholder("float",shape=[None,40], name='y_')

'''layer1'''
W_conv1 = weight_variable([3,3,1,20],name='W_conv1')
b_conv1 = bias_variable([20],name='b_conv1')
x_image = tf.reshape(x,[-1,57,47,1],name='x_image')
h_conv1 = conv2d(x_image, W_conv1,name='h_conv1')
h_add_conv1 = tf.add(h_conv1,b_conv1,name='h_add_conv1')
tanh_conv1 = tf.tanh(h_add_conv1,name='tanh_conv1')
h_pool1 = max_pool_2x2(tanh_conv1,name='h_pool1')#pool.shape=[-1,29,24,20]

'''layer2'''
W_conv2 = weight_variable([3,3,20,40],name='W_conv2')
b_conv2 = bias_variable([40],name='b_conv2')
h_conv2 = conv2d(h_pool1, W_conv2, name='h_conv2')
h_add_conv2 = tf.add(h_conv2,b_conv2,name='h_add_conv2')
tanh_conv2 = tf.tanh(h_add_conv2,name='tanh_conv2')
h_pool2 = max_pool_2x2(tanh_conv2,name='h_pool2')#pool.shape=[-1,15,12,40]

'''full connection'''
h_faltten = tf.reshape(h_pool2,[-1,15*12*40],name='h_faltten')
W_fc1 = weight_variable([15*12*40, 1000],name='W_fc1')
b_fc1 = bias_variable([1000],name='b_fc1')
h_fc1 = tf.matmul(h_faltten,W_fc1,name='h_fc1')
h_add_fc1 = tf.add(h_fc1,b_fc1,name='h_add_fc1')
h_tanh_fc1 = tf.tanh(h_add_fc1,name='h_tanh_fc1')

'''softmax'''
W_fc2 = weight_variable([1000,40],name='W_fc2')
b_fc2 = bias_variable([40],name='b_fc2')
h_fc2 = tf.matmul(h_tanh_fc1,W_fc2,name='h_fc2')
h_add_fc2 = tf.add(h_fc2,b_fc2,name='h_add_fc2')
h_softmax = tf.nn.softmax(h_add_fc2,name='h_softmax')

'''loss'''
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(h_softmax,1e-8,1.0)),name='cross_entropy')

'''train'''
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_softmax,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())
sess = tfdbg.LocalCLIDebugWrapperSession(sess)

for i in range(30000):
    print("start!!!!!!!!!!!!!!")
    batch_x,batch_y = bulider.next_batch_image(40)
    input()
    if i%1 == 0:
        train_cross_entropy = cross_entropy.eval(feed_dict={x:batch_x,y_:batch_y})
        print("step %d, cross entropy: %g"%(i, train_cross_entropy))
        #print(x_y_neighborhoodDiff.eval(feed_dict={x1:base, x2:target, y_:is_same}))
    train_step.run(feed_dict={x:batch_x,y_:batch_y})


batch_x,batch_y = bulider.next_batch_image(40)

print(sess.run(cross_entropy,feed_dict={x:batch_x,y_:batch_y}))
