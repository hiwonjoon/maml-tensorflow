from six.moves import xrange
import tensorflow as tf
from commons.ops import *

# Sinusoidal
def _sin_arch():
    net_spec = [Linear('linear_1',1,40),
                Lrelu(),
                Linear('linear_2',40,40),
                Lrelu(),
                Linear('linear_3',40,1),]
    weights = [block.get_variables() for block in net_spec]
    return net_spec, weights

def _sin_loss(logit,y):
    return tf.reduce_mean(tf.reduce_sum((logit - y)**2,axis=1)**0.5) #l2 loss for regression problem.

# Omniglot architecture
def _omniglot_arch(num_classes):
    net_spec = [Conv2d('conv2d_1',1,64,k_h=3,k_w=3),
                BatchNorm('conv2d_1',64,scale=False),
                lambda t,**kwargs : tf.nn.relu(t),
                Conv2d('conv2d_2',64,64,k_h=3,k_w=3),
                BatchNorm('conv2d_2',64,scale=False),
                lambda t,**kwargs : tf.nn.relu(t),
                Conv2d('conv2d_3',64,64,k_h=3,k_w=3),
                BatchNorm('conv2d_3',64,scale=False),
                lambda t,**kwargs : tf.nn.relu(t),
                Conv2d('conv2d_4',64,64,k_h=3,k_w=3),
                BatchNorm('conv2d_4',64,scale=False),
                lambda t,**kwargs : tf.nn.relu(t),
                lambda t,**kwargs : tf.reduce_mean(t,axis=[2,3]),
                Linear('linear',64,num_classes)] #N-way classificaiton
    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights

def _xent_loss(logits,labels,num_classes):
    #Note: sparse version does not have a second derivative..
    #return tf.reduce_mean(
    #    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))
    one_hots = tf.one_hot(labels, num_classes, axis=-1)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hots))

class Maml():
    def __init__(self,alpha,num_sgd,beta,global_step,
                 x,y,x_prime,y_prime,
                 arch_fn,loss_fn,
                 param_scope,is_training=False):
        with tf.variable_scope(param_scope):
            net_spec,weights = arch_fn()

        def _per_task(elems):
            x,y,x_prime,y_prime = elems
            # x, y is [Batch,InputDim], [Batch,OutputDim]
            # return grads of weights per task
            logits_per_steps = []
            task_weights = weights
            for it in range(num_sgd) :
                _t = x
                for block,ws in zip(net_spec,task_weights) :
                    _t = block(_t,is_training=True,**ws)
                task_loss = loss_fn(_t,y)

                task_weights = [
                    {key: w - alpha*tf.gradients(task_loss,[w])[0] for key,w in ws.items()} #maybe, calculating it as a batch might improve performance..
                    for ws in task_weights
                ]

                _t = x_prime
                for block,ws in zip(net_spec,task_weights) :
                    _t = block(_t,is_training=True,**ws) #update batch norm stats once.
                logits_per_steps.append(_t)

            loss = loss_fn(_t,y_prime)
            return (logits_per_steps[-1], loss, tf.stack(logits_per_steps,axis=-1))

        with tf.variable_scope('forward') as forward_scope:
            logits, loss, logits_per_steps = tf.map_fn(_per_task,[x,y,x_prime,y_prime],dtype=(tf.float32,tf.float32,tf.float32))
            self.logits = logits
            self.loss = tf.reduce_mean(loss)
            self.logits_per_steps = logits_per_steps #shape of logits + [NUM_SGD]


        if( is_training == False ):
            return

        with tf.variable_scope('backward'):
            optimizer = tf.train.AdamOptimizer(beta)
            grads = optimizer.compute_gradients(self.loss)
            self.train_op= optimizer.apply_gradients(grads,global_step=global_step)

            # We will only use on-the-fly statistics for batch norm.
            #with tf.variable_scope('bn_assign') as bn:
            #    print(x)
            #    _t = tf.reshape(x,[-1,1,28,28])
            #    print(_t)
            #    for block,ws in zip(net_spec,weights) :
            #        _t = block(_t,is_training=True,**ws)
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,bn.name)
            #print(update_ops)
            #with tf.control_dependencies(update_ops):
            #    grads = optimizer.compute_gradients(self.loss)
            #    self.train_op= optimizer.apply_gradients(grads,global_step=global_step)

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        #for name,var in save_vars.items():
        #    print(name,var)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)

if __name__ == "__main__":
    #with tf.variable_scope('params') as params :
    #    pass

    #global_step = tf.Variable(0, trainable=False)
    #alpha = tf.placeholder(tf.float32,shape=[])
    #beta = tf.placeholder(tf.float32,shape=[])
    #x = tf.placeholder(tf.float32,[None,None,1]) # TaskBatch, BatchNum, Input dim
    #y = tf.placeholder(tf.float32,[None,None,1])  # TaskBatch, BatchNum, Output dim
    #x_prime = tf.placeholder(tf.float32,[None,None,1]) # TaskBatch, BatchNum, Input dim
    #y_prime = tf.placeholder(tf.float32,[None,None,1])  # TaskBatch, BatchNum, Output dim

    #maml = Maml(alpha,beta,global_step,x,y,x_prime,y_prime,
    #            _sin_arch,_sin_loss,params)


    global_step = tf.Variable(0, trainable=False)
    alpha = tf.placeholder(tf.float32,shape=[])
    beta = tf.placeholder(tf.float32,shape=[])
    x = tf.placeholder(tf.float32,[None,None,1,28,28]) # TaskBatch, BatchNum, Input dim
    y = tf.placeholder(tf.int64,[None,None])  # TaskBatch, BatchNum, Output dim
    x_prime = tf.placeholder(tf.float32,[None,None,1,28,28]) # TaskBatch, BatchNum, Input dim
    y_prime = tf.placeholder(tf.int64,[None,None])  # TaskBatch, BatchNum, Output dim

    with tf.variable_scope('params') as params :
        pass

    from functools import partial
    net = Maml(alpha,2,beta,global_step,x,y,x_prime,y_prime,
               partial(_omniglot_arch,num_classes=20),partial(_xent_loss,num_classes=20),
               params,is_training=True)

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    from tqdm import tqdm
    import numpy as np
    for step in tqdm(xrange(100)):
        logits,loss,_ = sess.run([net.logits,net.loss,net.train_op],
                                 feed_dict={alpha:0.01,
                                            beta:0.01,
                                            x:np.random.uniform(0.0,1.0,(32,5,1,28,28)),
                                            y:np.random.randint(0,5,(32,5)),
                                            x_prime:np.random.uniform(0.0,1.0,(32,5,1,28,28)),
                                            y_prime:np.random.randint(0,5,(32,5))})
