# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from lib.model import CAE
from lib.utils import load_dict, save_image_test, save_image

class Sampler:
    def __init__(self, args):
        self.sample_nums = args.sample_nums
        self.cae = CAE(args)
        # create sample directory
        if not os.path.exists('data/sample'):
            os.makedirs('data/sample')
    
    def sample(self, visualized=True):
        # load checkpoint from disk
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='model/checkpoint')
        if ckpt is None:
            raise IOError('Checkpoint is not exist')
  
        self.cae.saver.restore(self.cae.session, ckpt.model_checkpoint_path)
        tf.get_variable_scope().reuse_variables()
        
        self.name_label_dict = load_dict('model/name_dict.txt')
        # sample   
        for (key,value) in self.name_label_dict.items():
            print "Sample person: {}".format(key)
            persons = []
            x_e_random1 = np.random.randint(0,self.cae.e_dim,self.sample_nums)
            x_e_random2 = np.random.randint(0,self.cae.e_dim,self.sample_nums)
            for idx in range(self.sample_nums):
                batch_persons = self.cae.generate_images(value, x_e_random1[idx], x_e_random2[idx])
                persons.append(batch_persons)
	    if visualized:
            	persons = np.array(persons)#.reshape(-1, 6, self.cae.x_dim,self.cae.x_dim,3)
            	save_image_test(persons, 'data/sample', key)
	    else:
		persons = np.array(persons).reshape(-1, self.cae.x_dim,self.cae.x_dim,3)
            	save_image_test(persons, 'data/sample', key)
