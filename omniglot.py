import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import Maml, _omniglot_arch, _xent_loss
from dataset import Omniglot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TASK_NUM,
         N_WAY,
         K_SHOTS,
         TRAIN_NUM,
         ALPHA,
         LEARNING_RATE, #BETA
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         SAVE_PERIOD,
         SUMMARY_PERIOD):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    omni = Omniglot()
    _,x,y,x_prime,y_prime= omni.build_queue(TASK_NUM,N_WAY,K_SHOTS)
    _,x_val,y_val,x_prime_val,y_prime_val = omni.build_queue(TASK_NUM,N_WAY,K_SHOTS,train=False)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        # Optimizing
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = Maml(ALPHA,learning_rate,global_step,x,y,x_prime,y_prime,
                   _omniglot_arch,_xent_loss,params,is_training=True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = Maml(ALPHA,0.0,tf.Variable(0,trainable=False),
                         x_val,y_val,x_prime_val,y_prime_val,
                         _omniglot_arch,_xent_loss,params,is_training=False)

    with tf.variable_scope('misc'):
        def _get_acc(logits,labels):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=-1),labels),tf.float32))

        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('acc',_get_acc(net.logits,y_prime))
        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        extended_summary_op = tf.summary.merge([
            tf.summary.scalar('valid_loss',valid_net.loss),
            tf.summary.scalar('valid_acc',_get_acc(valid_net.logits,y_prime_val)),
        ])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(TRAIN_NUM)):
            it,loss,_ = sess.run([global_step,net.loss,net.train_op])

            if( it % SAVE_PERIOD == 0 ):
                net.save(sess,LOG_DIR,step=it)

            if( it % SUMMARY_PERIOD == 0 ):
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary,it)
                tqdm.write('[%5d] Loss: %1.3f'%(it,loss))

            if( it % (SUMMARY_PERIOD*10) == 0 ): #Extended Summary
                summary = sess.run(extended_summary_op)
                summary_writer.add_summary(summary,it)

    except Exception as e:
        coord.request_stop(e)
    finally :
        net.save(sess,LOG_DIR)

        coord.request_stop()
        coord.join(threads)

def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/omniglot/%s'%(now),

        'TASK_NUM': 32,
        'N_WAY' : 5,
        'K_SHOTS': 1,

        'TRAIN_NUM' : 50000, #Size corresponds to one epoch
        'ALPHA': 0.4,
        'LEARNING_RATE' : 0.001,
        'DECAY_VAL' : 0.5,
        'DECAY_STEPS' : 25000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'SUMMARY_PERIOD' : 20,
        'SAVE_PERIOD' : 50000,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix

    main(config=config,**config)
