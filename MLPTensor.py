import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(.07, name='bias')

    z = w * x + b

    init = tf.global_variables_initializer()
    # Create a session and pass in graph g
    with tf.Session(graph=g) as sees:
        # Initilize w and b:
        sees.run(init)
        # Evaluate z:
        for t in [1.0, 0.6, -1.8]:
            print('x = %4.1f --> z = %4.1f' % (t, sees.run(z, feed_dict={x: t})))
