import numpy as np
import math
import time

start = time.time()
#Load required numpy arrays
masses = np.load('masses.npy')
positions = np.load('positions.npy')
velocities = np.load('velocities.npy')

#define global constants
G = 6.67e5
threshold = 0.1
time_step = 1e-4

def checkPositions(positions):
	for i in range(len(positions)):
		for j in range(i+1,len(positions)):
			if(np.linalg.norm(positions[i]-positions[j])<threshold):
				return True
	return False

count=0
while(checkPositions(positions)==False):
	new_positions = []
	new_velocities = []
	# minDist = 12000.0
	for i in range(len(positions)):
		accl = [0.0 , 0.0]
		for j in range(len(positions)):
			if(i==j):
				continue
			dist = np.linalg.norm(positions[i]-positions[j])
			accl[0] = accl[0]+(-G*masses[j][0]*(positions[i][0]-positions[j][0])/(dist**3))
			accl[1] = accl[1]+(-G*masses[j][0]*(positions[i][1]-positions[j][1])/(dist**3))
		new_positions.append([positions[i][0]+velocities[i][0]*time_step+(accl[0]*(time_step**2)/2),positions[i][1]+velocities[i][1]*time_step+(accl[1]*(time_step**2)/2)])
		new_velocities.append([velocities[i][0]+accl[0]*time_step,velocities[i][1]+accl[1]*time_step])
	positions = np.asarray(new_positions)
	velocities = np.asarray(new_velocities)
	count = count+1

end = time.time()
print 'No. of iterations:',count
print 'Execution time:',(end-start)
np.save('q2-normal-positions.npy',positions)
np.save('q2-normal-velocities.npy',velocities)