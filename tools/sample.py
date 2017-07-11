# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from lib.model import CAE
from lib.utils import load_dict, save_image_test, save_image

class Sampler:
    def __init__(self, args):
        self.cae = CAE(args)
        # create sample directory
        if not os.path.exists('data/sample'):
            os.makedirs('data/sample')
    
    def sample(self):
        # load checkpoint from disk
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='model/checkpoint')
        if ckpt is None:
            raise IOError('Checkpoint is not exist')
  
        self.cae.saver.restore(self.cae.session, ckpt.model_checkpoint_path)
        tf.get_variable_scope().reuse_variables()
        
        self.name_label_dict = load_dict('model/name_dict.txt')
	self.emotion_dict = load_dict('model/emotion_dict.txt')
        # sample   
	emotion_inverse = {}
	for (key,value) in self.emotion_dict.items():
	    emotion_inverse[int(value)] = key
	fusion_list = []
	for e1 in range(len(self.emotion_dict.keys())):
	    for e2 in range(e1+1,len(self.emotion_dict.keys())):
		fusion_list.append([e1,e2]) 
        for (key,value) in self.name_label_dict.items():
            print "Sample person: {}".format(key)
            for emo in fusion_list:
                batch_persons = self.cae.generate_images(value, emo[0], emo[1])
            	save_image_test(batch_persons, 'data/sample', '{}_{}_{}_mixed'.format(key,emotion_inverse[emo[0]],emotion_inverse[emo[1]]))
