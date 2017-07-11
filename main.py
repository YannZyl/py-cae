# -*- coding: utf-8 -*-
import argparse
import numpy as np
from lib.dataset import prepare_data
from lib.utils import load_dict
from tools.train import Trainer
from tools.sample import Sampler
from tools.align import align_face
        
parser = argparse.ArgumentParser()
parser.add_argument('--emotion_nums', type=int, default=7)
parser.add_argument('--transform_nums', type=int, default=4)
parser.add_argument('--person_nums', type=int, default=70)
parser.add_argument('--image_scale', type=int, default=173)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--l1_loss_weights', type=float, default=10.0)
parser.add_argument('--gan_loss_weights', type=float, default=0.01)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=1.0)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--lr_decay_steps', type=int, default=10000)
parser.add_argument('--max_step', type=int, default=200000)
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--save_step', type=int, default=1000)
parser.add_argument('--sampling', action='store_true')
parser.add_argument('--test_groups', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--generate_data',  action='store_true', help='If generate data again, including align and prepare')
parser.add_argument('--face_align',  action='store_true', help='Align face before prepare data')
parser.add_argument('--use_extra_data',  action='store_true', help='If use extra data for training')
parser.add_argument('--aligned_dir',  type=str, default='data/aligned_dataset', help='Aligned images directory')
parser.add_argument('--dataset_dir',  type=str, default='/home/zyl8129/Desktop/share_sample', help='Dataset directory')
parser.add_argument('--extra_data_dir',  type=str, default='/home/zyl8129/Documents/datasets/KDEF', help='Extra dataset directory')
args = parser.parse_args()

if __name__ == '__main__':
    
    np.random.seed(args.seed)

    if args.generate_data:
        # prepare datas, align images if necessary
        train_data_dir = args.dataset_dir
        if args.face_align:
           align_face(args.image_scale, args.dataset_dir, args.aligned_dir)
           #align_face(args.image_scale, args.extra_data_dir, args.aligned_dir)
           train_data_dir = args.aligned_dir
        # use extra data if necessary, but extra data not aligen
        if args.use_extra_data:
           datas = prepare_data(args.image_scale, train_data_dir, args.extra_data_dir)
        else:
           datas = prepare_data(args.image_scale, train_data_dir)
    # get person nums and emotion nums
    person_dict = load_dict('model/name_dict.txt')
    emotion_dict = load_dict('model/emotion_dict.txt')
    args.person_nums = len(person_dict.keys())
    args.emotion_nums = len(emotion_dict.keys())         
    if not args.sampling:
        t = Trainer(args)
        t.train()
    else:
        s = Sampler(args)
        s.sample()
