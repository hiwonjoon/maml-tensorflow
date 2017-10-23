from six.moves import xrange
import numpy as np
import tensorflow as tf

class Sinusoid():
    def __init__(self,
                 amp_range=(0.1,5.0),
                 phase_range=(0,np.pi),
                 input_range=(-5.0,5.0) ):
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.input_range = input_range

    def _generate_random_task(self):
        amp = np.random.uniform(*self.amp_range)
        phase = np.random.uniform(*self.phase_range)
        return amp, phase

    def _generate_random_task_batch(self,amp,phase,batch_size):
        x = np.random.uniform(*self.input_range, size=batch_size)
        y = amp * np.sin(x-phase)
        return x, y

    def _generate_valid_task_batch(self,amp,phase,num_pts):
        x  = np.linspace(*self.input_range,num=num_pts)
        y = amp * np.sin(x-phase)
        return x, y

    def build_queue(self,task_num,batch_size,train=True):
        # Actually, this does not build queue since data can generated on the fly.
        def _gen(task_num,batch_size,train):
            tasks = []
            xs,ys = [], []
            xs_prime, ys_prime = [], []
            for num in xrange(task_num):
                amp,phase = self._generate_random_task()
                x,y = self._generate_random_task_batch(amp,phase,batch_size)
                if(train):
                    x_prime,y_prime = self._generate_random_task_batch(amp,phase,batch_size)
                else:
                    x_prime,y_prime = self._generate_valid_task_batch(amp,phase,batch_size*20)
                tasks.append((amp,phase))
                xs.append(x)
                ys.append(y)
                xs_prime.append(x_prime)
                ys_prime.append(y_prime)
            return np.array(tasks,np.float32), \
                   np.array(xs,np.float32), np.array(ys,np.float32), \
                   np.array(xs_prime,np.float32), np.array(ys_prime,np.float32)

        tasks,x,y,x_prime,y_prime = tf.py_func(_gen,
                                               [task_num,batch_size,train],
                                               [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32],
                                               stateful=True)

        tasks = tf.reshape(tasks,[task_num,2])
        x = tf.reshape(x,[task_num,batch_size,1])
        y = tf.reshape(y,[task_num,batch_size,1])
        if( train ):
            x_prime = tf.reshape(x_prime,[task_num,batch_size,1])
            y_prime = tf.reshape(y_prime,[task_num,batch_size,1])
        else :
            x_prime = tf.reshape(x_prime,[task_num,batch_size*20,1])
            y_prime = tf.reshape(y_prime,[task_num,batch_size*20,1])

        return tasks,x,y,x_prime,y_prime

if __name__ == "__main__":
    from tqdm import tqdm
    sin = Sinusoid()

    gen_op = sin.build_queue(10,10)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for step in tqdm(xrange(100)):
            tasks,x,y,x_prime,y_prime= sess.run(gen_op)
    except Exception as e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

    print tasks
    print x
    print y
    print x_prime
    print y_prime

