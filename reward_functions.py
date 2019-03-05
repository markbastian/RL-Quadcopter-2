import numpy as np

def default_reward(task):
    """Uses current pose of sim to return reward."""
    reward = 1.-.3*(abs(task.sim.pose[:3] - task.target_pos)).sum()
    return reward

#This is the takeoff task
def rms_dist_fn(task):
    delta = task.sim.pose[:3] - task.target_pos
    return -np.sqrt(np.dot(delta, delta))

#With this reward function you would expect the agent to simply train for max thrust
def z_up_fn(task):
    return task.sim.pose[2]

def inverse_radius_fn(task):
    delta = task.sim.pose[0:3] - task.target_pos
    dist = np.linalg.norm(delta)
    if dist > 0.0:
        return 1.0 / dist
    else:
        return 0.0
    
def velocity_fn(task):
    delta = task.target_pos - task.sim.pose[0:3]
    dmag = np.linalg.norm(delta)
    if dmag == 0.0:
        #You are at the target, no reward
        return 0.0
    vel = task.sim.v
    vmag = np.linalg.norm(vel)
    if vmag == 0:
        return 0.0
    return np.dot(delta / dmag, vel / vmag)

# Suggestions from the first review
# You can add z-axis velocity in reward function to encourage quadcopter to fly towards the target.
# You can subtract angular velocity from the reward to make sure quadcopter flies straight up.
# You can include some large bonus and penalty rewards also. Such as a bonus on achieving the target height and a penalty on crashing.
# Clip your final reward between (-1, 1). It will definitely help in better performance.