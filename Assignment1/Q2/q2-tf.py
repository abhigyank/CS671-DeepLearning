import tensorflow as tf
import numpy as np
import time

start = time.time()
#Load required numpy arrays
massesArray = np.load('masses.npy')
positionsArray = np.load('positions.npy')
velocitiesArray = np.load('velocities.npy')

#define graph for the task
graph = tf.get_default_graph()

#define required constants
numParticles = massesArray.size
infinity = 1e20
threshold = 0.1
removeNan = tf.constant(1e-10,dtype='float64',name="removeNan")
G = tf.constant(-6.67e5,dtype='float64',name="G")
time_step = tf.constant(1e-4,dtype='float64',name="time_step")

#define required variables and masses
masses = tf.constant(massesArray,dtype='float64',name="masses")
velocities = tf.placeholder(dtype='float64',name="velocities")
positions = tf.placeholder(dtype='float64',name="positions")

#define required operations

#for finding updated acceleration
xpositions = tf.slice(positions,[0,0],[numParticles,1])
ypositions = tf.slice(positions,[0,1],[numParticles,1])
x_find_distances = tf.reshape(tf.math.add(tf.math.subtract(xpositions,tf.transpose(xpositions)),removeNan),[numParticles,numParticles])
y_find_distances = tf.reshape(tf.math.add(tf.math.subtract(ypositions,tf.transpose(ypositions)),removeNan),[numParticles,numParticles])
distances = tf.sqrt(tf.add(tf.square(x_find_distances),tf.square(y_find_distances)))
findDistances = tf.reduce_min(tf.matrix_set_diag(distances,tf.constant(infinity,dtype='float64',shape=[numParticles])))
x_vec_by_distance3 = tf.matrix_set_diag(tf.divide(x_find_distances,tf.pow(distances,3)),tf.zeros([numParticles],dtype='float64'))
y_vec_by_distance3 = tf.matrix_set_diag(tf.divide(y_find_distances,tf.pow(distances,3)),tf.zeros([numParticles],dtype='float64'))
vec_by_distance3 = tf.concat([tf.reshape(x_vec_by_distance3,[numParticles,numParticles,1]),tf.reshape(y_vec_by_distance3,[numParticles,numParticles,1])],2)
mult_dist_m = tf.multiply(vec_by_distance3,masses)
mult_distm_G = tf.multiply(tf.math.reduce_sum(mult_dist_m,1),G)
#for finding updated positions
mult_v_t = tf.multiply(velocities,time_step)
add_x_vt = tf.add(positions,mult_v_t)
mult_a_t2 = tf.multiply(mult_distm_G,tf.math.divide(tf.math.square(time_step),2.0))
add_xvt_at2 = tf.add(add_x_vt,mult_a_t2,name="updatePositions")
#for finding updated velocities
mult_a_t = tf.multiply(mult_distm_G,time_step)
add_v_at = tf.add(velocities,mult_a_t,name="updateVelocities")

init_op = tf.global_variables_initializer()
count = 0
with tf.Session() as sess:
	writer = tf.summary.FileWriter("./log",sess.graph)
	sess.run(init_op)
	while(1):
		count=count+1
		newpositionsArray = sess.run(add_xvt_at2,feed_dict={positions:positionsArray,velocities:velocitiesArray})
		velocitiesArray = sess.run(add_v_at,feed_dict={positions:positionsArray,velocities:velocitiesArray})
		positionsArray = newpositionsArray
		minDist = sess.run(findDistances,feed_dict={positions:positionsArray})
		# print 'Iteration#',count ,minDist
		if(minDist<threshold):
			break
	writer.close()

end = time.time()
print 'No. of iterations:',count
print 'Execution time:',(end-start)
np.save('q2-tf-positions.npy',positionsArray)
np.save('q2-tf-velocities.npy',velocitiesArray)
