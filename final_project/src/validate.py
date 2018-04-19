'''

----------------------------------------------------------------------
-- Load an saved model and validate it on validation set 
-- output the accuracy used by ECCV work shop

-- Jingxi Xu
----------------------------------------------------------------------

'''
from utilities import *
import argparse 
from model_LSTMSpatial import JingxiNet

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)

# set parameters for the model
class PARAMS(object):
    base_output_dir = "/home/jingxi/DL4CV_COMSW4995_006/final_project/log"
    # learning rate
    learningRate = 0.05
    # weightDecay = 5e-4
    learningRateDecayStep = 2500
    momentum = 0.9
    learningRateDecay = 0.96
    
    # hyper settings
    # batch size for validation is 100
    batch_size = 200
    forceNewModel = True
    targetScaleFactor = 1
    nGPUs = 1
    GPU = 1
    LSTM = True
    useCuda = True
    # 6000/128 * 10000
    nb_batches = 400000
    nb_show = 100
    nb_validate = 500
    nb_save = 1000


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, dest="saved_model", help="which saved model (no extension name) to load")
	args = parser.parse_args()

	# disable tensorflow debugging log
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	output_model_name = "JingxiNet"
	with tf.variable_scope(output_model_name, reuse=tf.AUTO_REUSE):
		#global step
		global_step = tf.Variable(0, trainable=False)
		jx_model = JingxiNet()
		jx_model.create_model()
		# prediction/frame_features/ground_truth: batch_size * 6 * 5
		loss = tf.reduce_sum(tf.square(jx_model.frame_features - jx_model.ground_truth)) / PARAMS.batch_size
		# prediction is 5-dimension so we need to mean twice
		accuracy_5d = tf.reduce_mean(tf.reduce_mean(tf.constant(1.) - tf.abs(jx_model.frame_features - jx_model.ground_truth), axis=0), axis=0)
		accuracy_1d = tf.reduce_mean(accuracy_5d)
		# learning rate
		lr = tf.train.exponential_decay(PARAMS.learningRate, global_step,PARAMS.learningRateDecayStep,PARAMS.learningRateDecay,staircase=True)
		# optimazation
		# did not use weight decay!!!!!
		train_jx = tf.train.MomentumOptimizer(lr, PARAMS.momentum).minimize(loss,global_step=global_step)

	with tf.Session() as sess:
		saver = tf.train.Saver(scope_variables(""))
		with tf.device('/gpu:0'):		
			saver.restore(sess, args.saved_model)
			print("Model {} loaded".format(args.saved_model))

		num_batch = len(validation_mp4_names)/PARAMS.batch_size
		# validation loss
		# validate for 10 times and then get average accuracy
		for i in range(1):
			results_1d = np.zeros(num_batch)
			results_5d = np.zeros((num_batch, 5))
			batch_index = 0
			for validation_set in get_next_validation_batch(PARAMS.batch_size):
				accuracy_1d_val, accuracy_5d_val = sess.run([accuracy_1d, accuracy_5d],
				                                  feed_dict={jx_model.audio_pl: validation_set['audio'],
				                                             jx_model.video_pl: validation_set['video'],
				                                             jx_model.ground_truth: validation_set['gt']})
				print("Overall accuracy: %.03f (Extraversion: %.03f, Agreeableness: %.03f, Conscientiousness: %.03f, Neurotisicm: %.03f, Openness: %.03f)" 
					% (accuracy_1d_val, accuracy_5d_val[0], accuracy_5d_val[1], accuracy_5d_val[2],
					accuracy_5d_val[3], accuracy_5d_val[4]))
				results_1d[batch_index] = accuracy_1d_val
				results_5d[batch_index] = accuracy_5d_val
				batch_index+=1

			# average over all batches
			average_1d = np.mean(results_1d)
			average_5d = np.mean(results_5d, axis=0)
			print("************ Summary *************")
			print("Overall accuracy: %.03f (Extraversion: %.03f, Agreeableness: %.03f, Conscientiousness: %.03f, Neurotisicm: %.03f, Openness: %.03f)" 
				% (average_1d, average_5d[0], average_5d[1], average_5d[2], average_5d[3], average_5d[4]))




  		