import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from cifar10 import *
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

weight_decay = 0.0005
momentum = 0.9
 
init_learning_rate = 0.1
 
batch_size = 60 
iteration = 782
# 128 * 391 ~ 50,000
test_iteration = 10
 
total_epochs = 1#100
 
def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = Relu(network)
        return network
  

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)
 
def Relu(x):
    return tf.nn.relu(x)                                                                                                                                                                                    
 
def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')
 
def Max_pooling(x, pool_size=[3,3], stride=2, padding='SAME') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_pooling_s(x, pool_size=[3,3], stride=1, padding='SAME'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
 
def Avg_pooling(x, pool_size=[3,3], stride=1, padding='VALID') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
 
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
 
def Concatenation(layers) :
    return tf.concat(layers, axis=3)
 
def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


#with tf.name_scope(scope):
#        x = tf.layers.average_pooling2d(inputs=x, pool_size=[5,5], stride=3, padding='VALID')
#        x = conv_layer(x, filter = 128, kernel=[1,1], strides=1, padding='VALID', layer_name=scope+'_conv1')
#        x = Fully_connected(x)
#        x = Fully_connected(x)
#        x = tf.nn.softmax(x)
#    return x
#    with tf.name_scope("pool2d"):
#       pooled = tf.layers.average_pooling2d(inputs, [5,5], stride=3, padding='VALID')
#    with tf.name_scope("conv11"):
#       conv11 = conv_layer(pooled, filter=52, kernel=[1,1], layer_name=scope+)
#    with tf.name_scope("flatten"):
#        flat = tf.reshape( conv11, [-1, 2048] )
#    with tf.name_scope("fc"):
#        fc = tf.layers.fully_connected( flat, 1024, activation_fn=None )
#    with tf.name_scope("drop"):
#        drop = tf.layers.dropout( fc, 0.3, is_training = is_training )
#    with tf.name_scope( "linear" ):
#        linear = tf.layers.fully_connected( drop, num_classes, activation_fn=None )
#    with tf.name_scope("soft"):
#        soft = tf.nn.softmax( linear )
#    return soft

def Evaluate(sess):                                                                                                                                                                                         
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000
 
    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add
 
        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }
 
        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)
 
        test_loss += loss_
        test_acc += acc_
 
    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy
 
    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
 
    return test_acc, test_loss, summary

class GoogLeNet():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_GoogLeNet(x)
    
    def Auxiliary_layer(self, input_x):
        x = Avg_pooling(input_x, pool_size=[5,5], stride=3)
        x = conv_layer(x, filter=256, kernel=[1,1])
        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.3, training=self.training)
        x = flatten(x)
        x = Fully_connected(x, layer_name='first_fully_connected')
        x = Fully_connected(x, layer_name='final_fully_connected')
        x = tf.nn.softmax(x)
        return x

    def Input_stem(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=[3,3], stride=1, layer_name=scope+'_conv1')
            x = Max_pooling_s(x)
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1') #LRN 대신 BN 집어넣음
            x = conv_layer(x, filter=64, kernel=[1,1], padding='VALID', layer_name=scope+'_conv2')
            x = conv_layer(x, filter=192, kernel=[3,3], layer_name=scope+'_conv3')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2') #LRN 대신 BN 집어넣음
            x = Relu(x)
            return x
 
    def Inception_3a(self, x, scope):
        with tf.name_scope(scope) :
            x = Max_pooling(x)
            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=96, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=128, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=16, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=32, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4 = Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=32, kernel=[1,1], layer_name=scope+'_split_conv6')
 
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
 
    def Inception_3b(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=96, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4 = Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=64, kernel=[1,1], layer_name=scope+'_split_conv6')
 
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
    def Inception_4a(self, x, scope):
        with tf.name_scope(scope) :
            x = Max_pooling_s(x)
            split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=96, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=208, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=16, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=64, kernel=[1,1], layer_name=scope+'_split_conv6')
 
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
 
    def Inception_4b(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=160, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=112, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=24, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=64, kernel=[1,1], layer_name=scope+'_split_conv6')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x

    def Inception_4c(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=24, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=64, kernel=[1,1], layer_name=scope+'_split_conv6')
            
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
 
    def Inception_4d(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=112, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=144, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=64, kernel=[1,1], layer_name=scope+'_split_conv6')
            
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x

    def Inception_4e(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=160, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=320, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=128, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=128, kernel=[1,1], layer_name=scope+'_split_conv6')
            #Aux_4e = Auxiliary_layer(x)
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
  
    def Inception_5a(self, x, scope):
        with tf.name_scope(scope) :
            x = Max_pooling(x)
            split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=160, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=320, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=128, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=128, kernel=[1,1], layer_name=scope+'_split_conv6')
 
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
    
    def Inception_5b(self, x, scope):
        with tf.name_scope(scope) :
            init = x
            split_conv_x1 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x2 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = Batch_Normalization(split_conv_x2, training=self.training, scope=scope+'_batch1')
            split_conv_x2= Relu(split_conv_x2)
            split_conv_x2 = conv_layer(split_conv_x2, filter=384, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(x, filter=48, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = Batch_Normalization(split_conv_x3, training=self.training, scope=scope+'_batch2')
            split_conv_x3= Relu(split_conv_x3)
            split_conv_x3 = conv_layer(split_conv_x3, filter=128, kernel=[5,5], layer_name=scope+'_split_conv5')
            split_conv_x4 = Max_pooling_s(x)
            split_conv_x4 = Batch_Normalization(split_conv_x4, training=self.training, scope=scope+'_batch3')
            split_conv_x4= Relu(split_conv_x4)
            split_conv_x4 = conv_layer(split_conv_x4, filter=128, kernel=[1,1], layer_name=scope+'_split_conv6')
            
            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])
 
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch4')
            x= Relu(x)
 
            return x
   
    def Output_stem(self, x, scope):
        with tf.name_scope(scope) :
            x = Avg_pooling(x)
            x = Relu(x)
        return x 
 
    def Build_GoogLeNet(self, input_x):
        #input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # size 32 -> 96
        # only cifar10 architecture
 
        L1 = self.Input_stem(input_x, scope='Input_stem'+str(1))
        print(L1.get_shape())
        tf.add_to_collection('layer:1', L1)
        L2 = self.Inception_3a(L1, scope='Inception_3a'+str(2))
        print(L2.get_shape())
        L3 = self.Inception_3b(L2, scope='Inception_3b'+str(3))
        print(L3.get_shape())
        L4 = self.Inception_4a(L3, scope='Inception_4a'+str(4))
        print(L4.get_shape())
        L5 = self.Inception_4b(L4, scope='Inception_4b'+str(5))
        print(L5.get_shape())
        tf.add_to_collection('aux:0', L5)
        A4b = self.Auxiliary_layer(L4)
        #print("Aux:0 shape : ", A4b.get_shape())
        L6 = self.Inception_4c(L5, scope='Inception_4c'+str(6))
        print(L6.get_shape())
        L7 = self.Inception_4d(L6, scope='Inception_4d'+str(7))
        print(L7.get_shape())
        L8 = self.Inception_4e(L7, scope='Inception_4e'+str(8))
        print(L8.get_shape())
        tf.add_to_collection('aux:1', L8)
        #print("Aux:0 shape : ", A4b.get_shape())
        A4e = self.Auxiliary_layer(L7)
        #tf.add_to_collection('aux:1', A4e)
        L9 = self.Inception_5a(L8, scope='Inception_5a'+str(9))
        print(L9.get_shape())
        L10 = self.Inception_5b(L9, scope='Inception_5b'+str(10))
        print(L10.get_shape())
        L11 = self.Output_stem(L10, scope='Output_stem'+str(11))
        print(L11.get_shape())
        tf.add_to_collection('output', L11)
        L12 = Global_Average_Pooling(L11)
        L13 = Dropout(L12, rate=0.3, training=self.training)
        L14 = flatten(L13)
        print(L14.get_shape())
        L15 = Fully_connected(L14, layer_name='final_fully_connected')
        print(L15.get_shape())

        L16 = tf.nn.softmax(L15)
        return L16, A4b, A4e #L16:Softmax_2:0, A4b:Softmax:0, A4e:Softmax_1:0

train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)
 
 
# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
 
training_flag = tf.placeholder(tf.bool)
 
 
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
 
model = GoogLeNet(x, training=training_flag).model
#print(model[0])
loss_o = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model[0], labels=label))
loss_a1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model[1], labels=label))
loss_a2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model[2], labels=label))
cost = loss_o + 0.3*(loss_a1 + loss_a2)

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
 
correct_prediction = tf.equal(tf.argmax(model[0], 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = False 
with tf.Session(config=config) as sess:

    #ckpt = tf.train.get_checkpoint_state('./model')
    #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #   saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(tf.global_variables_initializer())
 
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    epoch_learning_rate = init_learning_rate
    for epoch in range(total_epochs):
        if epoch % 20 == 0 :
            epoch_learning_rate = epoch_learning_rate / 3
        if epoch % 30 == 0 :
            epoch_learning_rate = epoch_learning_rate / 3
 
        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
 
        for step in range(1, iteration + 1):
            if pre_index + batch_size < 50000:
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
            else:
                batch_x = train_x[pre_index:]
                batch_y = train_y[pre_index:]
 
            #batch_x = data_augmentation(batch_x)
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            aux0 = tf.get_collection('aux:0')
            aux1 = tf.get_collection('aux:1')
            output = tf.get_collection('output')

            _, batch_loss, aux0_value, aux1_value, output_ = sess.run([train, cost, aux0, aux1, output], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)
 
            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size
 
 
        train_loss /= iteration # average loss
        train_acc /= iteration # average accuracy
 
        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
 
        test_acc, test_loss, test_summary = Evaluate(sess)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()
 
        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)
        
        np.save('aux/aux0_%d' %epoch, aux0_value)
        np.save('aux/aux1_%d' %epoch, aux1_value)
        np.save('aux/output_%d' %epoch, output_)

        with open('logs.txt', 'a') as f:
            f.write(line)
 
        saver.save(sess=sess, save_path='./model/Inception_v4.ckpt')





