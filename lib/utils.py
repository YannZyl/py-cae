# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import numpy as np

# save/load name-label dictionary
def save_dict(filename, saved_dict):
    f = open(filename, 'w')  
    pickle.dump(saved_dict, f)  
    f.close()
    
def load_dict(filename):
    f = open(filename, 'r') 
    loaded_dict = pickle.load(f)
    f.close() 
    return loaded_dict
    
def next_batch(datas, offset, batch_size, im_size):
    pid, eid, tid, im_list = datas
    max_batch = len(im_list) // batch_size
    offset = offset % max_batch
    batch_pid = pid[offset*batch_size:(offset+1)*batch_size]
    batch_eid = eid[offset*batch_size:(offset+1)*batch_size]
    batch_tid = tid[offset*batch_size:(offset+1)*batch_size]
    batch_imlist = im_list[offset*batch_size:(offset+1)*batch_size]
    batch_ims = []
    for name in batch_imlist:
        im = cv2.imread(name).astype(np.float32)
        im = cv2.resize(im, (im_size,im_size)).astype(np.float32)
        im = im / 127.5 - 1.0
        batch_ims.append(im)
    batch_ims = np.array(batch_ims)
    return [batch_pid, batch_eid, batch_tid, batch_ims]

# save images during training step
def save_image(images, columns, save_dir, global_step):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    shape = images.shape
    rows = shape[0] // columns
    saves = np.zeros((rows*shape[1], columns*shape[2], shape[3]))
    for idx in range(shape[0]):
        row = idx // columns
        col = idx % columns
        saves[row*shape[1]:(row+1)*shape[1],col*shape[2]:(col+1)*shape[2],:] = images[idx]
    saves = np.clip((saves+1.0)*127.5,0,255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir,'generated_{}.png'.format(global_step)), saves)

# save images during testing step(after finish training)
def save_image_test(images, save_dir, person_name):
    images = np.clip((images+1.0)*127.5,0,255).astype(np.uint8)
    if not os.path.exists(os.path.join(save_dir, person_name)):
        os.mkdir(os.path.join(save_dir, person_name))
    if len(images.shape) == 4:
    	for idx, image in enumerate(images):
            cv2.imwrite(os.path.join(save_dir,person_name,'sampled_{}.png'.format(idx+1)), image)
    else:
	n, g, h, w, c = images.shape
        saves = np.zeros((n*h, g*w, c))
        for nid in range(n):
	    image = images[nid]
	    for gid in range(g):
 		saves[nid*h:(nid+1)*h, gid*w:(gid+1)*w, :] = image[gid]
        cv2.imwrite(os.path.join(save_dir,person_name,'interpolation_{}.png'.format(person_name)), saves)
	    
