import numpy as np
from physics_sim import PhysicsSim

class Env():
    def __init__(self, 
                 init_pose=np.array([0., 0., 0., 0., 0., 0.]), 
                 init_velocities=np.array([0., 0., 0.]), 
                 init_angle_velocities=np.array([0., 0., 0.]), 
                 runtime=5.0, 
                 target_pos=None,
                 reward_fn=None):
        self.reward_fn = reward_fn
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.target_pos = target_pos
        self.state_size = 6
        self.action_size = 4
        self.action_low = 0
        self.action_high = 900
        
    def step(self, rotor_speeds):
        #You MUST do the sim step BEFORE computing the reward.
        done = self.sim.next_timestep(rotor_speeds)
        reward = self.reward_fn(self)
        return self.sim.pose, reward, done, {}

    def reset(self):
        self.sim.reset()
        return self.sim.pose