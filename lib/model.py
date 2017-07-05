# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import tensorflow as tf
from layers import *
from lib.inputs import data_loader

class GCN:
    def __init__(self, args):
        self.p_dim = args.person_nums
        self.e_dim = args.emotion_nums
        self.t_dim = args.transform_nums
        self.x_dim = args.image_scale 
        self.batch_size = args.batch_size
        self.euclidean_loss_weights = args.euclidean_loss_weights
        self.person_loss_weights = args.person_loss_weights
        self.emotion_loss_weights = args.emotion_loss_weights
        self.transform_loss_weights = args.transform_loss_weights
        self.learning_rate = args.learning_rate
        self.lr_low_boundary = 1e-5
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.lr_decay_steps = args.lr_decay_steps
        
        self.loss_gan_weight = 0.01
        self.loss_l1_weight = 0.99
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.build_model()
        
    def build_model(self):
        # inputs and labels
        with tf.variable_scope('input_labels'):
            # load data via tensorflow queue
            x = data_loader(self.x_dim,self.p_dim,self.e_dim,self.t_dim,self.batch_size)
            self.y = x[0]
            self.x_p = x[1]
            self.x_e = x[2]
            self.x_t = x[3]         
        # network outputs 
        with tf.name_scope('train_step'):
            self.gen_image = self.generator(self.x_p, self.x_e, self.x_t)
            predict_real, pid_real, eid_real, tid_real = self.discriminator(self.y, self.x_p, self.x_e, self.x_t)
            predict_fake, pid_fake, eid_fake, tid_fake = self.discriminator(self.gen_image, self.x_p, self.x_e, self.x_t, reuse=True)          
        # network outputs 
        with tf.name_scope('sample_step'):
            self.s_p = tf.placeholder(tf.float32, [None,self.p_dim])
            self.s_e = tf.placeholder(tf.float32, [None,self.e_dim])
            self.s_t = tf.placeholder(tf.float32, [None,self.t_dim])
            self.sam_image = self.generator(self.s_p, self.s_e, self.s_t, reuse=True)
        with tf.name_scope('public_loss'):
            pid_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pid_real,labels=self.x_p))
            pid_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pid_fake,labels=self.x_p))
            eid_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=eid_real,labels=self.x_e))
            eid_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=eid_fake,labels=self.x_e))
            tid_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tid_real,labels=self.x_t))
            tid_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tid_fake,labels=self.x_t))
        # discriminator loss
        with tf.name_scope('d_loss'):
            discrim_loss = self.lambda1*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_real,labels=tf.ones_like(predict_real)))+\
                           self.lambda1*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake,labels=tf.zeros_like(predict_fake)))+\
                           self.lambda2*(pid_loss_real + pid_loss_fake + eid_loss_real + eid_loss_fake + tid_loss_real + tid_loss_fake)              
        # generator loss
        with tf.name_scope('g_loss'):            
            gen_loss_GAN = self.lambda1*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake,labels=tf.ones_like(predict_fake)))+\
                           self.lambda2*(pid_loss_fake + eid_loss_fake + tid_loss_fake)
            gen_loss_L1 = tf.reduce_mean(tf.abs(self.y-self.gen_image))
            gen_loss = gen_loss_GAN * self.loss_gan_weight  + gen_loss_L1 * self.loss_l1_weight 
        # discriminator optimizer
        with tf.name_scope('d_optimizer'):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
        # generator optimizer
        with tf.name_scope('g_optimizer'):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                g_lr = tf.Variable(self.learning_rate, trainable=False)
                gen_optim = tf.train.AdamOptimizer(g_lr, self.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
        # loss average move
        with tf.name_scope('average_move'):
            with tf.control_dependencies([gen_train]):
                ema = tf.train.ExponentialMovingAverage(decay=0.99)
                update_loss = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
        # update learning rate 
        with tf.name_scope('lr_update'):        
            g_lr_update = tf.assign(g_lr, tf.maximum(g_lr*0.5, 0.00001))

        self.fetch_dict = {
            'discrim_loss': ema.average(discrim_loss),
            'gen_loss_gan': ema.average(gen_loss_GAN),
            'gen_loss_l1': ema.average(gen_loss_L1)
        }

        self.train_items = update_loss
        self.lr_update = g_lr_update
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        
    def generate_images(self, identify, emotion1, emotion2=None, interpolate=True):
        x_p, x_e, x_t = [], [], []
        # first people condition
        x_p1 = [0] * self.p_dim; x_p1[identify] = 1; x_p.append(x_p1)     
        x_e1 = [0] * self.e_dim; x_e1[emotion1] = 1; x_e.append(x_e1)      
        x_t1 = [1] + [0]*(self.t_dim-1); x_t.append(x_t1)
        # if use interpolate via the second person, build fusion condition 
        if interpolate:
            # second people condition
	    scale_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
            x_e2 = [0] * self.e_dim; x_e2[emotion2] = 1
	    for ratio in scale_ratios:    
            	# fusion condition
            	x_p.append(x_p1)
            	x_e3 = [ratio*x_e2[idx]+(1-ratio)*x_e1[idx] for idx in range(self.e_dim)]
            	x_e.append(x_e3)
            	x_t.append(x_t1)
        # change to ndarray
        x_p = np.array(x_p)
        x_e = np.array(x_e)
        x_t = np.array(x_t)
        # generate
        persons = self.session.run(self.sam_image, {self.s_p:x_p, self.s_e:x_e, self.s_t:x_t})
        return persons
    
    # generator1, default
    def generator(self, person, emotion, transform, reuse=None):
        with tf.variable_scope('generator', reuse=reuse):
            # branch1: person
            x_person = add_fc(person, 256, activation_fn=None)
            x_person = add_lrelu(x_person, 0.1)
            x_person = add_fc(x_person, 512, activation_fn=None)
            # branch2: emotion
            x_emotion = add_fc(emotion, 256, activation_fn=None)
            x_emotion = add_lrelu(x_emotion, 0.1)
            x_emotion = add_fc(x_emotion, 512, activation_fn=None)
            # branch3: transform
            x_transform = add_fc(transform, 128, activation_fn=None)
            x_transform = add_lrelu(x_transform, 0.1)
            x_transform = add_fc(x_transform, 256, activation_fn=None)
            # merged 3 branches
            x = tf.concat([x_person,x_emotion,x_transform], axis=1)
            x = add_lrelu(x, 0.1)
            
            # 3 fully connected layers
            fc_layers_spec = [2048, 2048, 8*8*512]
            for idx, output_nums in enumerate(fc_layers_spec):
                x = add_fc(x, output_nums, activation_fn=None)
                x = add_lrelu(x, 0.1)
                if idx == len(fc_layers_spec)-1:
                    x = tf.reshape(x, [-1,8,8,512])     
            # 4 convolutional transpose layers
            deconv_layers_spec = [512, 184, 184, 3]
            for idx, output_nums in enumerate(deconv_layers_spec):
                x = add_deconv2d(x, output_nums, 5, 2, 'VALID', activation_fn=None)
                if idx < len(deconv_layers_spec)-1:
                    x = add_lrelu(x, 0.1)     
            return add_tanh(x)

    # discriminator(pix2pix net)
    def discriminator(self, inputs, person, emotion, transform, reuse=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            n_layers = 3
            ndf = 64
            # branch1: person
            x_person = add_fc(person, 256, activation_fn=None)
            # branch2: emotion
            x_emotion = add_fc(emotion, 256, activation_fn=None)
            # branch3: transform
            x_transform = add_fc(transform, 128, activation_fn=None)
            # merged 3 branches
            x = tf.concat([x_person,x_emotion,x_transform], axis=1)
            x = add_lrelu(x, 0.2)
            
            _, h, w, c = inputs.get_shape().as_list()
            x_2 = add_fc(x, h*w, activation_fn=None)
            x_2 = add_lrelu(x_2, 0.2)
            x_2 = tf.reshape(x_2, [-1,h,w,1])
            x = tf.concat([inputs,x_2], axis=3)
            
            # layer_1: [batch, 173, 173, in_channels] => [batch, 86, 86, ndf]
            x = add_conv2d(x, ndf, 3, 2, 'VALID', activation_fn=None)
            x = add_lrelu(x, 0.2)
            
            # layer_2: [batch, 86, 86, ndf] => [batch, 43, 43, ndf * 2]
            # layer_3: [batch, 43, 43, ndf * 2] => [batch, 22, 22, ndf * 4]
            # layer_4: [batch, 22, 22, ndf * 4] => [batch, 11, 11, ndf * 8]
            for i in range(n_layers):
                out_channels = ndf*(2**(i+1))
                x = add_conv2d(x, out_channels, 3, 2, 'SAME', activation_fn=None)
                x = add_batchnorm(x, is_training=True)
                x = add_lrelu(x, negatice_slop=0.2)
            
            # layer_5: 
            _, h, w, c = x.get_shape().as_list()
            x = tf.reshape(x, [-1,h*w*c])
            x = add_fc(x, 1024, activation_fn=None)
            x = add_lrelu(x, 0.2)
            
            # real/fake output
            x1 = add_fc(x, 1, activation_fn=None)
            # person output
            x2 = add_fc(x, self.p_dim, activation_fn=None)
            # emotion output
            x3 = add_fc(x, self.e_dim, activation_fn=None)
            # transform output
            x4 = add_fc(x, self.t_dim, activation_fn=None)
            return x1, x2, x3, x4