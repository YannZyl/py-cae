# -*- coding: utf-8 -*-
import cv2
from glob import glob
import tensorflow as tf

def to_onehot(indice, depth, on_value=1, off_value=0):
    feat = [off_value] * depth
    feat[indice] = on_value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=feat))
    
def generate_tfrecoder(p_dim, e_dim, t_dim):
    # get file names
    filename = glob('data/images/*')
    # write tfrecoder
    writer = tf.python_io.TFRecordWriter('data/train.tfrecords')
    for pfile in filename:
        name = (pfile.split('/')[-1]).split('.')[0]
        pid, eid, tid, _ = name.split('_')
        image = cv2.imread(pfile)
        raw_image = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'pid': to_onehot(int(pid), p_dim),
            'eid': to_onehot(int(eid), e_dim),
            'tid': to_onehot(int(tid), t_dim),
            'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_image]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def data_loader(x_dim, p_dim, e_dim, t_dim, batch_size):
    # get filename queue
    generate_tfrecoder(p_dim, e_dim, t_dim)
    
    filename_queue = tf.train.string_input_producer(['data/train.tfrecords'])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    single_example = tf.parse_single_example(serialized_example,
                                       features={
                                            'pid': tf.FixedLenFeature([p_dim], tf.int64),
                                            'eid': tf.FixedLenFeature([e_dim], tf.int64),
                                            'tid': tf.FixedLenFeature([t_dim], tf.int64),
                                            'raw_image': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(single_example['raw_image'], tf.uint8)
    img = tf.reshape(img, [x_dim, x_dim, 3])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    pid = tf.cast(single_example['pid'], tf.float32) 
    eid = tf.cast(single_example['eid'], tf.float32)  
    tid = tf.cast(single_example['tid'], tf.float32)
    
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    img_batch, pid_batch, eid_batch, tid_batch = tf.train.shuffle_batch([img, pid, eid, tid],
                                                                        batch_size=batch_size, capacity=capacity, 
                                                                        min_after_dequeue=min_after_dequeue)

    return img_batch, pid_batch, eid_batch, tid_batch

  
        