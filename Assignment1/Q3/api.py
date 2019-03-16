import tensorflow as tf

def dense(inps, input_dim, output_dim, bias_constant=0.1, activation=None):
    weights =  tf.Variable(tf.truncated_normal([input_dim, output_dim]))
    biases = tf.Variable(tf.constant(bias_constant, shape = [output_dim]))
    layer = tf.add(tf.matmul(inps, weights), biases)
    if(activation == None):
        return layer,weights
    else:
        return activation(layer), weights