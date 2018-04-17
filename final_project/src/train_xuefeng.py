
# coding: utf-8

# In[18]:


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(suppress=True)
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


import socket
HOSTNAME = socket.gethostname()
print('Running on'+ HOSTNAME)
if(HOSTNAME != "pineapple"):
    raise ValueError("Please run on pineapple")


# ## Get a batch of data

# In[20]:



train_target_file = '/data/junting/DL4CV_COMSW4995_006/final_project/training_gt.csv'
validation_target_file = '/data/junting/DL4CV_COMSW4995_006/final_project/validation_gt.csv'

# first load ground truth into a numpy file
train_gt_np = np.genfromtxt(train_target_file, delimiter=',', dtype=object)
validation_gt_np = np.genfromtxt(validation_target_file, delimiter=',', dtype=object)

# get a list of mp4 names
train_mp4_names = train_gt_np[1:, 0]
validation_mp4_names = validation_gt_np[1:, 0]

# construct a dictionary for ground truth
train_gt = {}
validation_gt = {}
for i in range(1,train_gt_np.shape[0]):
    name = train_gt_np[i, 0]
    scores = train_gt_np[i, 1:].astype(float)
    train_gt[name] = scores
for i in range(1,validation_gt_np.shape[0]):
    name = validation_gt_np[i, 0]
    scores = validation_gt_np[i, 1:].astype(float)
    validation_gt[name] = scores


# In[21]:


train_frames_path = '/data/junting/ECCV/trainframes'
validation_frames_path = '/data/junting/ECCV/validationframes'

train_audiofeat_path = '/data/junting/ECCV/trainaudiofeat/'
validation_audiofeat_path = '/data/junting/ECCV/validationaudiofeat/'

def get_next_batch(batch_size):
    batch = {}
    length = len(train_mp4_names)
    train_names=np.random.permutation(train_mp4_names)
    epoch_count = 0
    batch_count = 0
    start = 0
    while True:
        # randomly choose a list of mp4 names to consist the batch
        if start >= length or start + batch_size > length:
            train_index = np.random.shuffle(train_index)
            train_names = np.random.permutation(train_mp4_names)
            epoch_count += 1
            start = 0
            batch_count = 0
        mp4_names = train_names[start: start + batch_size]
        batch_count += 1
        # get video batches -> batch_size * 6 * 112 * 112 * 3
        # video = np.zeros((batch_size, 6, 112, 112, 3))
        pre_time = time.time()
        vid_names = np.array(mp4_names)
        with Parallel(n_jobs=batch_size/4) as parallel:
            frames = parallel(delayed(load_vid_img)(vid_name)
                              for vid_name in vid_names)
            video = np.asarray(frames)
        print("epoch:{}, batch{}, load frames use: {}s".format(epoch_count, batch_count, time.time()-pre_time))
                # get audio batches -> batch_size * 6 * 68
        audio = np.zeros((batch_size, 6, 68))
        for batch_num, mp4 in enumerate(mp4_names):
            audiofeat_name = mp4+'.wav.csv'
            audio[batch_num] = np.genfromtxt(os.path.join(train_audiofeat_path, audiofeat_name), delimiter=',')

        # get ground truth -> batch_size * 6 * 5 (copied 5 times)
        gt = np.zeros((batch_size, 6, 5))
        for batch_num, mp4 in enumerate(mp4_names):
            gt[batch_num] = np.tile(np.array(train_gt[mp4]), (6, 1))

        batch['gt'] = gt
        batch['audio'] = audio
        batch['video'] = video
        start += batch_size
        yield batch

def load_vid_img(mp4):
    all_frames = os.listdir(os.path.join(train_frames_path, mp4.replace('.mp4', '')))
    num_frames = len(all_frames)  # int
    interval = num_frames / 6
    frames=[]
    for i in range(0, 6):
        index = i * interval + np.random.randint(1, interval + 1)
        frame_name = 'frame_det_00_%06d.png' % (index)
        frame = mpimg.imread(os.path.join(train_frames_path, mp4.replace('.mp4', ''), frame_name))
        frames.append(frame)
    return frames


# ## Make the model

# In[22]:


class PARAMS(object):
    base_output_dir = "/data/junting/DL4CV_COMSW4995_006/final_project/models"
    #learning rate
    learningRate = 0.05
    #weightDecay = 5e-4
    learningRateDecayStep = 2500
    momentum = 0.9
    learningRateDecay = 0.96
    
    #hyper settings
    batch_size = 128
    forceNewModel = True
    targetScaleFactor = 1
    nGPUs = 1
    GPU = 1
    LSTM = True
    useCuda = True
    #6000/128 * 10000
    nb_batches = 400000
    nb_show = 100
    nb_validate = 500
    nb_save = 1000
    


# In[23]:


def get_logdirs_and_modelname(PARAMS):
    """Set log directories and model names."""
    log_output_dir = "log"
    log_output_dir += "_" + str(PARAMS.learningRate)
    log_output_dir += "_" + str(PARAMS.learningRateDecayStep)
    log_output_dir += "_" + str(PARAMS.learningRateDecay)
    output_dir = os.path.join(PARAMS.base_output_dir, log_output_dir)
    output_model_name = "JingxiNet"
    return output_dir, output_model_name

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)

# In[25]:


if __name__ == "__main__":
    import time
    from model_LSTMSpatial import JingxiNet
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, dest="gpu", default=3, help="gpu num")
    args = parser.parse_args()
    output_log_dir, output_model_name = get_logdirs_and_modelname(PARAMS)

    with tf.variable_scope(output_model_name, reuse=tf.AUTO_REUSE):
        #global step
        global_step = tf.Variable(0, trainable=False)
        jx_model = JingxiNet()
        jx_model.create_model()
        #time stamp 
        start_ts = time.time()
        #loss
        #loss = tf.nn.l2_loss(jx_model.frame_features - jx_model.ground_truth)
        loss = tf.reduce_sum(tf.square(jx_model.frame_features - jx_model.ground_truth)) / PARAMS.batch_size
        #learning rate
        lr = tf.train.exponential_decay(PARAMS.learningRate, global_step,PARAMS.learningRateDecayStep,PARAMS.learningRateDecay,staircase=True)
        #optimazation
        #did not use weight decay!!!!!
        train_jx = tf.train.MomentumOptimizer(lr, PARAMS.momentum).minimize(loss,global_step=global_step)

        with tf.device('/gpu:3'):
            sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            sess.run(tf.global_variables_initializer())
            print("Graph defined and initialized in {}s.".format(time.time() - start_ts))

        #cpu work
        with tf.device('/cpu:0'):
            training_summary = tf.summary.scalar("training loss", loss)
            validation_summary = tf.summary.scalar("validation loss", loss)
            #learning_rate = tf.summary.scalar("learning_rate", lr)
            all_summary = tf.summary.merge_all()

    print(scope_variables(""))
    saver = tf.train.Saver(scope_variables(""))
    train_writer = tf.summary.FileWriter(output_log_dir, sess.graph)
    #start training
    count_batch = 0
    #JINGXI: please return current_batch to be a structure such as:
    #current_batch = {'audio': audio_feat, 'image': image_feat, 'gt': ground_truth}
    for current_batch in get_next_batch(PARAMS.batch_size):
        #run session
        loss_val, _ = sess.run([loss,train_jx], feed_dict={jx_model.audio_pl: current_batch['audio'],
                                      jx_model.video_pl: current_batch['video'],
                                      jx_model.ground_truth: current_batch['gt']})
        print("step %d/%d: train loss: %f" % (count_batch, PARAMS.nb_batches, loss_val))
        #show result
        if count_batch % PARAMS.nb_show == 0:
            #train loss
            train_summ, train_loss = sess.run([training_summary, loss],
                                              feed_dict={jx_model.audio_pl: current_batch['audio'],
                                                         jx_model.video_pl: current_batch['video'],
                                                         jx_model.ground_truth: current_batch['gt']})
            train_writer.add_summary(train_summ, count_batch)
            print("step %d/%d: validation loss: %f" % (count_batch, PARAMS.nb_batches,train_loss))

        #if count_batch % PARAMS.nb_validate == 0:
        #    #validation loss
        #    validation_summ, validation_loss = sess.run([validation_summary, loss],
        #                                      feed_dict={jx_model.audio_pl: validation_set['audio'],
        #                                                 jx_model.video_pl: validation_set['image'],
        #                                                 jx_model.ground_truth: validation_set['gt']})
        #    train_writer.add_summary(validation_summ, count_batch)
        #    print("step %d/%d: validation loss: %f" % (count_batch, PARAMS.nb_batches,validation_loss))

        #save model
        if count_batch % PARAMS.nb_save == 0:
            saver.save(sess, os.path.join(PARAMS.base_output_dir, output_model_name+'_'+str(count_batch)))

        count_batch += 1
        #end condition
        if count_batch > PARAMS.nb_batches:
            break

    # Save final model
    saver.save(sess, os.path.join(PARAMS.base_output_dir, output_model_name))

    
    #TODO: validation not done

