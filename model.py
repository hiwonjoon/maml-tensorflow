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

class Maml():
    def __init__(self,alpha,beta,global_step,
                 x,y,x_prime,y_prime,
                 arch_fn,loss_fn,
                 param_scope):
        with tf.variable_scope(param_scope):
            net_spec,weights = arch_fn()

        def _per_task(elems):
            x,y,x_prime,y_prime = elems
            # x, y is [Batch,InputDim], [Batch,OutputDim]
            # return grads of weights per task
            with tf.variable_scope(None,'per_task') :
                _t = x
                for block in net_spec :
                    _t = block(_t)
                task_loss = _sin_loss(_t,y)

                task_weights = [
                    {key: w + alpha*tf.gradients(task_loss,[w])[0] for key,w in ws.items()} #maybe, calculating it as a batch might improve performance..
                    for ws in weights
                ]
                _t = x_prime
                for block,ws in zip(net_spec,task_weights) :
                    _t = block(_t,**ws)
                logits = _t
                loss = loss_fn(_t,y_prime)
            return logits, loss

        with tf.variable_scope('forward'):
            logits, loss = tf.map_fn(_per_task,[x,y,x_prime,y_prime],dtype=(tf.float32,tf.float32))
            self.logits = logits #shape of x_prime
            self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('backward'):
            optimizer = tf.train.AdamOptimizer(beta)
            grads = optimizer.compute_gradients(loss,var_list=[w for ws in weights for _,w in ws.items()]) #flatten..
            self.train_op= optimizer.apply_gradients(grads,global_step=global_step)

        save_vars = {('train/'+'/'.join(var.name.split('/')[1:])).split(':')[0] : var for var in
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,param_scope.name) }
        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 3)

    def save(self,sess,dir,step=None):
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,sess,model):
        self.saver.restore(sess,model)

if __name__ == "__main__":
    with tf.variable_scope('params') as params :
        pass

    alpha = tf.placeholder(tf.float32,shape=[])
    beta = tf.placeholder(tf.float32,shape=[])
    x = tf.placeholder(tf.float32,[None,None,1]) # TaskBatch, BatchNum, Input dim
    y = tf.placeholder(tf.float32,[None,None,1])  # TaskBatch, BatchNum, Output dim
    x_prime = tf.placeholder(tf.float32,[None,None,1]) # TaskBatch, BatchNum, Input dim
    y_prime = tf.placeholder(tf.float32,[None,None,1])  # TaskBatch, BatchNum, Output dim

    maml = Maml(alpha,beta,x,y,x_prime,y_prime,
                _sin_arch,_sin_loss,params)
