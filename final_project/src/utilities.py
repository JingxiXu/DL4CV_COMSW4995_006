import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging
import time
from joblib import Parallel, delayed



train_target_file = '/data/junting/DL4CV_COMSW4995_006/final_project/training_gt.csv'
validation_target_file = '/data/junting/DL4CV_COMSW4995_006/final_project/validation_gt.csv'

# first load ground truth into a numpy file
train_gt_np = np.genfromtxt(train_target_file, delimiter=',', dtype=object)
validation_gt_np = np.genfromtxt(validation_target_file, delimiter=',', dtype=object)

# get a full list of mp4 names
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


train_frames_path = '/data/junting/ECCV/trainframes'
validation_frames_path = '/data/junting/ECCV/validationframes'

train_audiofeat_path = '/data/junting/ECCV/trainaudiofeat/'
validation_audiofeat_path = '/data/junting/ECCV/validationaudiofeat/'

# get a batch of video, audio and ground truth for training
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

# get the set for validation
def get_validation_set():
    print("Producing Validation Set")
    validation_set = {}
    
    # to construct validation set, use all videos
    mp4_names = validation_mp4_names

    # get video batches -> 2000 * 6 * 112 * 112 * 3
    video = np.zeros((2000, 6, 112, 112, 3))
    for batch_num, mp4 in enumerate(mp4_names):            
        all_frames = os.listdir(os.path.join(validation_frames_path, mp4.replace('.mp4', '')))
        num_frames = len(all_frames) # int
        interval = num_frames/6
        for i in range(0, 6):
            index = i*interval + np.random.randint(1, interval+1)
            frame_name = 'frame_det_00_%06d.png' % (index)
            frame = mpimg.imread(os.path.join(validation_frames_path, mp4.replace('.mp4', ''), frame_name))
            video[batch_num, i] = frame

    # get audio batches -> 2000 * 6 * 68
    audio = np.zeros((2000, 6, 68))
    for batch_num, mp4 in enumerate(mp4_names):
        audiofeat_name = mp4+'.wav.csv'
        audio[batch_num] = np.genfromtxt(os.path.join(validation_audiofeat_path, audiofeat_name), delimiter=',')

    # get ground truth -> 2000 * 6 * 5 (copied 5 times)
    gt = np.zeros((2000, 6, 5))
    for batch_num, mp4 in enumerate(mp4_names):
        gt[batch_num] = np.tile(np.array(validation_gt[mp4]), (6, 1))

    validation_set['gt'] = gt
    validation_set['audio'] = audio
    validation_set['video'] = video
    
    np.save("/data/junting/ECCV/validation_set/run1_gt.npy", gt)
    np.save("/data/junting/ECCV/validation_set/run1_audio.npy", audio)
    np.save("/data/junting/ECCV/validation_set/run1_video.npy", video)
    print("Validation Set Saved!")

    return validation_set

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

# to create the parent directory (recursively if the fiolder structure is not existing already)
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise
