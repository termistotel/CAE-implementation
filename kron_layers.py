# Zhou, Shuchang, Jia-Nan Wu, Yuxin Wu, and Xinyu Zhou. 2015. “Exploiting Local Structures with the Kronecker Layer in Convolutional Networks,” December. https://arxiv.org/abs/1512.09194.

import tensorflow as tf

def nmode(x, A, i):
	broj = i
	rank = len(x.get_shape().as_list())
	permut = list(range(rank))
	permut = permut[:broj] + permut[-1:] + permut[broj:-1]

	t = tf.tensordot(x, A, [[broj], [1]])
	return tf.transpose(t, permut)

def kfc(x, shapesA, shapesB, test=False):
	# Kroneker
	out = tf.zeros(shape=tf.shape(x))
	for shapeA, shapeB in zip(shapesA, shapesB):
		nodes = shapeA[0] * shapeB[0]
		A = tf.Variable(tf.random.normal(shape=shapeA))
		B = tf.Variable(tf.random.normal(shape=shapeB))
		bb = tf.Variable(tf.zeros(shape=(nodes)))

		z = tf.reshape(x, (-1, shapeA[1], shapeB[1]))
		z = nmode(z, A, 1)
		z = nmode(z, B, 2)
		out += tf.reshape(z, (-1, nodes)) + bb

	return out


def kfcPic(x, outshape, r = 1):
	tmpshape = tf.shape(x)
	# inshape = tmpshape[1:]
	inshape = x.get_shape().as_list()[1:]

	# for i = 0
	# A = tf.Variable(tf.random.normal(shape=(outshape[0],inshape[0])))
	# B = tf.Variable(tf.random.normal(shape=(outshape[1],inshape[1])))
	# C = tf.Variable(tf.random.normal(shape=(outshape[2],inshape[2])))
	# out = nmode(nmode(nmode(x, A, 1), B, 2), C, 3)
	out = 0
	for i in range(r):
		A = tf.Variable(tf.random.normal(shape=(outshape[0],inshape[0])))
		B = tf.Variable(tf.random.normal(shape=(outshape[1],inshape[1])))
		C = tf.Variable(tf.random.normal(shape=(outshape[2],inshape[2])))
		out += nmode(nmode(nmode(x, A, 1), B, 2), C, 3)

	print("tu1")
	bb = tf.Variable(tf.zeros(shape=outshape))
	print("tu2")
	return out+bb
