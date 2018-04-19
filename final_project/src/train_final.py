
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging
import time

import datetime
now = datetime.datetime.now()


from model_LSTMSpatial import JingxiNet
from utilities import *

# print suppressed float
np.set_printoptions(suppress=True)

import argparse
from tqdm import tqdm

import socket
HOSTNAME = socket.gethostname()
print('Running on'+ HOSTNAME)
if(HOSTNAME != "pineapple"):
    raise ValueError("Please run on pineapple")


###### Make the model
class PARAMS(object):
    base_output_dir = "/data/junting/DL4CV_COMSW4995_006/final_project/log"
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


def get_logdirs_and_modelname(PARAMS):
    """Set log directories and model names."""
    tensorboard_dir = "tensorboard"
    tensorboard_dir += "_" + str(PARAMS.learningRate)
    tensorboard_dir += "_" + str(PARAMS.learningRateDecayStep)
    tensorboard_dir += "_" + str(PARAMS.learningRateDecay)

    log_output_dir = "JingxiLog_%d_%02d_%02d_%02d_%02d_%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    log_output_dir = os.path.join(PARAMS.base_output_dir, log_output_dir)

    tensorboard_dir = os.path.join(PARAMS.base_output_dir, log_output_dir, tensorboard_dir)
    
    output_model_name = "JingxiNet"
    return log_output_dir, tensorboard_dir, output_model_name

def scope_variables(name):
    with tf.variable_scope(name):
        print("shit: ", tf.get_variable_scope().name)
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, dest="gpu", default=0, help="which GPU device to use")
    args = parser.parse_args()

    # the program can only see this particular GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # disable tensorflow debugging log
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    ## log_output_dir is the overall folder for log info
    ## tensorboard_dir is the sub-folder for tensorboard log
    ## output_model_name = "JingxiNet"
    log_output_dir, tensorboard_dir, output_model_name = get_logdirs_and_modelname(PARAMS)
    ## validation_set = get_validation_set()
    with tf.variable_scope(output_model_name, reuse=tf.AUTO_REUSE):
        #global step
        global_step = tf.Variable(0, trainable=False)
        jx_model = JingxiNet()
        jx_model.create_model()
        #time stamp 
        start_ts = time.time()
        # loss
        # loss = tf.nn.l2_loss(jx_model.frame_features - jx_model.ground_truth)
        loss = tf.reduce_sum(tf.square(jx_model.frame_features - jx_model.ground_truth)) / PARAMS.batch_size
        # learning rate
        lr = tf.train.exponential_decay(PARAMS.learningRate, global_step,PARAMS.learningRateDecayStep,PARAMS.learningRateDecay,staircase=True)
        # optimazation
        # did not use weight decay!!!!!
        train_jx = tf.train.MomentumOptimizer(lr, PARAMS.momentum).minimize(loss,global_step=global_step)

        with tf.device('/gpu:' + str(args.gpu)):
            sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            print("Graph defined and initialized in {}s.".format(time.time() - start_ts))

        #cpu work
        with tf.device('/cpu:0'):
            training_summary = tf.summary.scalar("training loss", loss)
            validation_summary = tf.summary.scalar("validation loss", loss)
            #learning_rate = tf.summary.scalar("learning_rate", lr)
            all_summary = tf.summary.merge_all()

    print("scope_variables: ", scope_variables(""))
    saver = tf.train.Saver(scope_variables(""))

    # config tensorboard
    train_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    ### start training
    count_batch = 0
    for current_batch in get_next_batch(PARAMS.batch_size):
        # run session
        train_summ, loss_val, _ = sess.run([training_summary, loss,train_jx], feed_dict={jx_model.audio_pl: current_batch['audio'],
                                      jx_model.video_pl: current_batch['video'],
                                      jx_model.ground_truth: current_batch['gt']})
        print("step %d/%d: train loss: %f" % (count_batch, PARAMS.nb_batches, loss_val))
        # show result
        if count_batch % PARAMS.nb_show == 0:
            #log train loss
            train_writer.add_summary(train_summ, count_batch)

        # save model
        if count_batch % PARAMS.nb_save == 0:
            saver.save(sess, os.path.join(log_output_dir, output_model_name+'_'+str(count_batch)))

        count_batch += 1
        # end condition
        if count_batch > PARAMS.nb_batches:
            break

    # Save final model
    saver.save(sess, os.path.join(log_output_dir, output_model_name + '_final'))


