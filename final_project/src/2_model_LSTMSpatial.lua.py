'''

----------------------------------------------------------------------
-- data loading (audio, video frames)

-- charlie
----------------------------------------------------------------------

'''

import tensorflow as tf
from ops import *

class bi_convlstm_net(object):
    def __init__(self):
        self.nstates = {64, 15, 5}
        self.filtrowsize = {91, 1}
        self.filtcolsize = {39, 5}
        self.rowstrides = {30, 1}
        self.colstrides = {1, 2}
        self.audio_dim = 68

    def create_model(self):

        self.audio_pl = tf.placeholder(tf.float32, [-1, 6, 68], name="audio_pl")
        self.video_pl = tf.placeholder(tf.float32, [-1, 6, 112, 112, 3], name="audio_pl")

        # -- -1 * 6 x 68 --> -1 * 6 x 32
        audio_branch = tf.reshape(self.audio_pl,[-1, 68])
        audio_branch = linear(audio_branch, 32, "au_fc1")
        # -- -1 * 6 * 112x112x3  --> -1 * 112x112x3
        video_branch = tf.reshape(self.video_pl,[-1, 112, 112, 3])
        # -- -1 * 112x112x3  --> -1 * 108x108x16
        video_branch = relu(conv2d(video_branch, 16, 5, 5, name = "vd_conv1", padding="VALID"))
        # -- -1 * 108x108x16 --> -1 * 54x54x16
        video_branch = MaxPooling(video_branch, 2, stride=2)

        # -- -1 * 54x54x16  --> -1 * 48x48x16
        video_branch = relu(conv2d(video_branch, 16, 7, 7, name="vd_conv2", padding="VALID"))
        # -- -1 * 48x48x16 --> -1 * 24x24x16
        video_branch = MaxPooling(video_branch, 2, stride=2)

        # -- -1 * 24x24x16  --> -1 * 16x16x16
        video_branch = relu(conv2d(video_branch, 16, 9, 9, name="vd_conv3", padding="VALID"))
        # -- -1 * 16x16x16 --> -1 * 8x8x16
        video_branch = MaxPooling(video_branch, 2, stride=2)

        # -1 * 8x8x16 -> -1 * 1024
        video_branch = tf.reshape(video_branch, [-1, 8*8*16])
        # -1 x 1024 --> -1 x 128
        video_branch = relu(linear(video_branch, 128, "vd_fc1"))

        # -1 * 32 + 128 -> -1 * 160
        frame_features = tf.concat([audio_branch, video_branch], 1)
        frame_features = tf.nn.dropout(frame_features, 1-0.2)
        frame_features = tf.reshape(frame_features, [-1, 6, 160])
        for_cell = tf.nn.rnn_cell.LSTMCell(num_units = 160, num_proj = 128, state_is_tuple=True)
        # [batch_size, 6, 128]
        frame_features, state = tf.nn.dynamic_rnn(for_cell, frame_features, dtype=frame_features.dtype)
        frame_features = tf.reshape(frame_features, [-1, 128])
        frame_features = tf.nn.dropout(frame_features, 1-0.2)
        frame_features = tf.sigmoid(linear(frame_features, 5, "fea_fc1"))
        self.frame_features = tf.reshape(frame_features, [-1, 6, 5])



