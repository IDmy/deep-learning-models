import tensorflow as tf
import argparse
import utils
import re
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import random

parser = argparse.ArgumentParser()

# I/O related args
parser.add_argument('--input_path', default='man_tshirts/', type=str,
                    help='Input path to your images.')
                    
parser.add_argument('--output_path', default='dcgan_out_man_tshirts/', type=str,
                    help='Output path for the generated images.')

parser.add_argument('--log_path', default='dcgan_tensorboard_log/', type=str,
                    help='Log path for tensorboard.')

# hyper-parameters
parser.add_argument('--z_dim', default=100, type=int,
                    help='Output path for the generated images.')

parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size used for training.')

parser.add_argument('--train_steps', default=10000, type=int,
                    help='Number of steps used for training.')
                    
FLAGS = parser.parse_args()
                    
# Image related constants
HEIGHT = 64
WIDTH = 64
DIM = 3

FLAT_DIM = HEIGHT * WIDTH * DIM

tf.reset_default_graph()
batch_size = FLAGS.batch_size
n_noise = FLAGS.z_dim

X_in = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, DIM], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.z_dim])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


#============================= DESCRIMINATOR ==================================
def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, HEIGHT, WIDTH, DIM])
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
#        x = tf.layers.conv2d(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
#        x = tf.layers.dropout(x, keep_prob)
#        x = tf.layers.conv2d(x, kernel_size=5, filters=1024, strides=2, padding='same', activation=activation)
#        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=512, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x

#============================= GENERATOR ======================================
def generator(z, keep_prob=keep_prob):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 512 #1024
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, decay=momentum)  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[8, 8]) #size=[4, 4]
#        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
#        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.sigmoid)
        return x

#========================== Loss functions and optimizers =====================       
g = generator(noise, keep_prob)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)
        
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#=============================== Load the dataset =============================

images_dir = FLAGS.input_path
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
labels = []
images_df = []

for k in tqdm(list_images):
    img = image.load_img(k, target_size=(HEIGHT, WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)    
    image_id = k.split('/')[-1].split('.')[0]
    labels.append(image_id)
    images_df.append(x)
    
images_arr = np.array(images_df)
images_arr = np.reshape(images_df, (len(list_images), HEIGHT, WIDTH, DIM))

#============================== Training ======================================
# create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# create summary writer
summary_writer = tf.summary.FileWriter(FLAGS.log_path, graph=tf.get_default_graph())

for i in tqdm(range(FLAGS.train_steps)):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 
    
    n = np.random.uniform(0.0, 1.0, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)   
    batch = np.array(random.sample(list(images_arr), FLAGS.batch_size))       
    
    if i % 500 == 0:
        samples = sess.run(g, feed_dict={noise: n})
        utils.save_plot(samples, FLAGS.output_path, i)
    
    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train})
    
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls
    
    if g_ls * 1.5 < d_ls:
        train_g = False
        pass
    if d_ls * 2 < g_ls:
        train_d = False
        pass
    
    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train})
        
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train})
        
# eventually print train losses
    if i % 500 == 0:
       print('Iter: {}'.format(i))
       print('D loss: {:.4}'.format(d_ls))
       print('G_loss: {:.4}'.format(g_ls))
       print()








