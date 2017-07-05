# -*- coding: utf-8 -*-
import os
import cv2
from utils import save_dict
from scipy.ndimage.interpolation import rotate

name_dict = {}
id_index = 0
emos_dict = {}

# perpare data from KDEF dataset
def prepare_data_kdef(im_size, data_dir, valid_nums=None):
    global id_index, name_dict
    # Expression:
    #    AF = afraid, AN = angry, DI = disgusted, HA = happy, NE = neutral, SA = sad, SU = surprised    
    emos = {'AN':0, 'HA':1, 'NE':2, 'SA':3, 'SU':4, 'AF':5, 'DI':6}
    # get each person's dir
    for idx, folder in enumerate(os.listdir(data_dir)):
        # choose certain perple for training 
        if valid_nums is not None and idx >= valid_nums:
            break
        
        # update dictionary
        label = folder[1:]
        if not name_dict.has_key(label):
            num_label = id_index
            name_dict[label] = id_index
            id_index += 1
        else:
            num_label = name_dict[label]
            
        # generate data
        for pic in os.listdir(data_dir+'/'+folder):
            
            if not pic.endswith('.JPG'):
                continue
            
            name = pic.split('.')[0]
            emotion = name[4:6]
            position = name[6:]
            pic_idx = 0 if name[0]=='A' else 1
            
            if position == 'S':
                # image read
                im = cv2.imread(os.path.join(data_dir,folder,pic))
                # crop
                im = im[200:600,100:500,:]
                im = cv2.resize(im, (im_size,im_size))
                angles = [0,90,180,270]
                for ang_idx, angle in enumerate(angles):
                    rotate_im = rotate(im, angle)
                    image_name = 'data/images/{}_{}_{}_{}.jpg'.format(num_label, emos(emotion), ang_idx, pic_idx)
                    cv2.imwrite(image_name, rotate_im)
    # update emos
    emos_dict.update(emos)
    
    
# perpare data from self dataset    
def prepare_data_self(im_size, data_dir):
    global id_index, name_dict
    exts = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG'] 
    emos = {'angry':0, 'happy':1, 'neutral':2, 'sad':3, 'surprise':4}
    # get each person's dir
    for folder in os.listdir(data_dir):
        
        name, emotion = folder.split('_')
        if emotion not in emos.keys():
            print 'Illegal emotion:{}, corresponding folders:{}'.format(emotion, folder)
            continue
        
        # read image and process image
        for pic_idx, pic in enumerate(os.listdir(data_dir+'/'+folder)):
            # check picture format and choose '202'(frontal) image
            if not pic.split('.')[-1] in exts or not pic.split('-')[1].startswith('202'):
                continue
            
            # update name-label dictionary and is label [name ==> number]
            if not name_dict.has_key(name):
                num_label = id_index
                name_dict[name] = id_index
                id_index += 1
            else:
                num_label = name_dict[name]   
            
            # prepare data
            im = cv2.imread(os.path.join(data_dir,folder,pic))
            im = cv2.resize(im, (im_size,im_size))
            # generate 4 angles 
            angles = [0,90,180,270]
            #angles = [0]
            for ang_idx, angle in enumerate(angles):
                rotate_im = rotate(im, angle)
                image_name = 'data/images/{}_{}_{}_{}.jpg'.format(num_label, emos[emotion], ang_idx, pic_idx)
                cv2.imwrite(image_name, rotate_im)
    # update dictionary
    emos_dict.update(emos)
    

def prepare_data(im_size, data_dir, extra_data_dir=False):
    # create directory if not exist
    if not os.path.exists('data/images'):
        os.makedirs('data/images')
        
    # prepare self data
    prepare_data_self(im_size, data_dir)
    # if use extra data, add kdef data into list
    if extra_data_dir:
        prepare_data_kdef(im_size, extra_data_dir, 70)
        
    # save name-label dictionary
    print 'Use extra data: {}, #persons: {}, #emotions: {}'.format(extra_data_dir,len(name_dict.keys()),len(emos_dict.keys()))
    save_dict('model/name_dict.txt', name_dict)
    save_dict('model/emotion_dict.txt', emos_dict)


if __name__ == '__main__':
    prepare_data(173, '/home/zyl8129/Desktop/share_sample', None)
    print name_dict
    print emos_dict