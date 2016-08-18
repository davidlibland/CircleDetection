#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime


log_dir = 'CircleDetectLog'
num_dots = 128
batch_size = 128
hidden_size = 64
keep_prob = 0.5
num_training_batches = 5000
learning_rate = 0.001

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2./shape[0]))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
x = tf.placeholder(tf.float32,[None,num_dots,2],"dots")
y_ = tf.placeholder(tf.float32,[None,1],name = "circle_present")
keep_prob_pl = tf.placeholder(tf.float32,name='keep_prob')


with tf.variable_scope('network'):
    w_1 = weight_variable([num_dots*2,hidden_size])
    b_1 = bias_variable([hidden_size])

    layer_1 = tf.nn.relu(tf.matmul(tf.reshape(x,[-1,num_dots*2]),w_1)+b_1)
    layer_1_drop = tf.nn.dropout(layer_1,keep_prob_pl)

    w_2 = weight_variable([hidden_size,hidden_size])
    b_2 = bias_variable([hidden_size])

    layer_2 = tf.nn.relu(tf.matmul(layer_1_drop,w_2)+b_2)
    layer_2_drop = tf.nn.dropout(layer_2,keep_prob_pl)

    w_3 = weight_variable([hidden_size,hidden_size])
    b_3 = bias_variable([hidden_size])

    layer_3 = tf.nn.relu(tf.matmul(layer_2_drop,w_3)+b_3)
    layer_3_drop = tf.nn.dropout(layer_3,keep_prob_pl)

    w_4 = weight_variable([hidden_size,1])
    b_4 = bias_variable([1])

    logits = tf.matmul(layer_3_drop,w_4)+b_4
    #logits = tf.matmul(layer_1_drop,w_4)+b_4 #Uncomment for single hidden layer.
    y = tf.nn.sigmoid(logits)


with tf.variable_scope('training_model'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits,y_,name="cross_entropy_loss"))
    tf.scalar_summary('cross_entropy_error',cross_entropy_loss)
    model_lr = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(model_lr)
    # the gradient descent objective incorporates our priors via L2 regularization to compute the MAP
    train = optimizer.minimize(cross_entropy_loss)

merged = tf.merge_all_summaries()
time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
train_writer = tf.train.SummaryWriter(os.path.join(log_dir,'train'+time_stamp),
                                        tf.get_default_graph())
valid_writer = tf.train.SummaryWriter(os.path.join(log_dir,'valid'+time_stamp))

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for step in range(num_training_batches):
        # generate the training samples:
        targets = np.random.randint(0,2,[batch_size,1])
        inputs_sq = np.random.normal(size = [batch_size,num_dots,2])
        inputs_deg = np.random.uniform(0,2*np.pi,size = [batch_size,num_dots])
        inputs_circ = np.stack([np.cos(inputs_deg),np.sin(inputs_deg)],axis=-1)
        # choose circles for target 1 and squares for target zero.
        inputs = targets.reshape([-1,1,1])*inputs_circ + (1-targets.reshape([-1,1,1]))*inputs_sq
        # non-linearly transform the data
        transforms = np.random.normal(scale = 5.,size = [batch_size,2,2])
        inputs = np.einsum('ijk,ikl->ijl',inputs,transforms)
        summary,training_loss,_ = sess.run([merged,cross_entropy_loss,train],
                                       {x:inputs,
                                        y_:targets,
                                       model_lr:learning_rate,
                                       keep_prob_pl:keep_prob})
        
        txt = 'Step %d of %d, training loss: %.4f'%(step,num_training_batches,training_loss)
        print(txt)
#         with open(os.path.join(log_dir,"output.txt"),'a') as f_out:
#                 f_out.write('\n'+txt)
        train_writer.add_summary(summary, step)
        if step%10 == 0 :
            train_writer.flush()
        if step%500 == 0 :
            plt.scatter(x = inputs[0,:,0],y = inputs[0,:,1])
            plt.title('Actual: %d, Predicted %f'%(targets[0,0],y.eval({x:inputs[0:1,:,:],keep_prob_pl:1.})[0]))
            plt.show()