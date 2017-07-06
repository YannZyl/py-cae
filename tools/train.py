# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from tqdm import trange
from lib.model import CAE
from lib.utils import save_image

class Trainer:
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.extra_data_dir = args.extra_data_dir
        self.aligned_dir = args.aligned_dir
        self.face_align = args.face_align
        self.use_extra_data = args.use_extra_data
        self.test_groups = args.test_groups
        self.max_step = args.max_step
        self.log_step = args.log_step
        self.save_step = args.save_step
        self.cae = CAE(args)
        # create output directory
        if not os.path.exists('data/output'):
            os.makedirs('data/output')  
        # create checkpoint directory
        if not os.path.exists('model/checkpoint'):
            os.makedirs('model/checkpoint')
    
    # interpolate flag:
    #   if True: emotion1, emotion2, 0.5*emotion1+0.5*emotion2
    #   if False: emotion1 only
    def sample_internal(self, step, interpolate=True):     
        persons = []
        x_p_random  = np.random.randint(0,self.cae.p_dim,self.test_groups)
        # interpolate: False, each emotion generate
        x_e_orders  = [range(self.cae.e_dim)]*self.test_groups
        # interpolate: True, choose 5 pair emotions <random1, random2> to interpolate ti systhesize new image
        x_e_random1 = np.random.randint(0,self.cae.e_dim,self.test_groups)
        x_e_random2 = np.random.randint(0,self.cae.e_dim,self.test_groups)
        for i in range(self.test_groups):
            if interpolate:
                batch_persons = self.cae.generate_images(x_p_random[i], x_e_random1[i], x_e_random2[i], interpolate)
                persons.append(batch_persons)
            else:
                for emotions in x_e_orders[i]:
                    batch_persons = self.cae.generate_images(x_p_random[i], emotions, None, interpolate)
                    persons.append(batch_persons)
        
        persons = np.array(persons).reshape(-1,self.cae.x_dim,self.cae.x_dim,3)
        columns = 3 if interpolate else self.cae.e_dim
        save_image(persons, columns, 'data/output', step)
   
    def train(self):
        # initialize global variables
        with self.cae.session.as_default():
            self.cae.session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(coord=coord)
            
            for step in trange(self.max_step):
                # run optimizer
                self.cae.session.run(self.cae.train_items)
                if step % self.log_step == 0:
                    results = self.cae.session.run(self.cae.fetch_dict)
                    print 'Iterations:{}, d_loss:{}, g_loss_gan:{}, g_loss_l1:{}'.format(step, \
                            results['discrim_loss'], results['gen_loss_gan'], results['gen_loss_l1'])
                if step % self.save_step == 0:
                    self.sample_internal(step, interpolate=False)
                    self.cae.saver.save(self.cae.session, 'model/checkpoint/cae.ckpt')
                if step % self.cae.lr_decay_steps == 0 and step !=0:
                    #self.cae.session.run(self.cae.lr_update)
                    pass
                
            coord.request_stop()
            coord.join(thread)
