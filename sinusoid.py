from six.moves import xrange
import better_exceptions
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model import Maml, _sin_arch, _sin_loss
from dataset import Sinusoid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TASK_NUM,
         BATCH_SIZE,
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
    sin = Sinusoid()
    _,x,y,x_prime,y_prime= sin.build_queue(TASK_NUM,BATCH_SIZE)
    _,x_val,y_val,x_prime_val,y_prime_val = sin.build_queue(TASK_NUM,BATCH_SIZE,train=False)
    # <<<<<<<

    # >>>>>>> MODEL
    with tf.variable_scope('train'):
        # Optimizing
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = Maml(ALPHA,1,learning_rate,global_step,
                   x,y,x_prime,y_prime,
                   _sin_arch,_sin_loss,params,is_training=True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = Maml(ALPHA,1,0.0,tf.Variable(0,trainable=False),
                         x_val,y_val,x_prime_val,y_prime_val,
                         _sin_arch,_sin_loss,params)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('valid_loss',valid_net.loss),

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        # Plot summary
        def _py_draw_plot(train_x,train_y,x,gt,pred):
            fig, ax = plt.subplots()
            ax.plot(np.squeeze(train_x),np.squeeze(train_y), 'ro')
            ax.plot(np.squeeze(x),np.squeeze(gt),'r')
            ax.plot(np.squeeze(x),np.squeeze(pred),'b')
            ax.set_ylim([-5.0,5.0])
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()

        def _tf_draw_plots(elems):
            train_x,train_y,x,gt,pred = elems
            png_str = tf.py_func(_py_draw_plot,
                                 [train_x,train_y,x,gt,pred],
                                 tf.string,
                                 stateful=False)
            return tf.image.decode_png(png_str, channels=4)

        plots = tf.map_fn(_tf_draw_plots,[x_val[:10],y_val[:10],
                                          x_prime_val[:10],y_prime_val[:10],
                                          valid_net.logits[:10]],dtype=tf.uint8)
        plots = tf.stack(plots,axis=0)
        # Expensive Summaries
        extended_summary_op = tf.summary.merge([
            tf.summary.image('results',plots,max_outputs=10)
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
        'LOG_DIR':'./log/sinusoid/%s'%(now),

        'TASK_NUM': 25,
        'BATCH_SIZE' : 20,

        'TRAIN_NUM' : 50000, #Size corresponds to one epoch
        'ALPHA': 0.01,
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
