import numpy as np
import gym


class PendulumTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = gym.make('Pendulum-v0')
        self.action_repeat = 3

        self.state_size = self.action_repeat * 3
        self.action_low = -2
        self.action_high = 2
        self.action_size = 1

        self.max_steps = 300
        self.current_steps = 0

        # Goal
        self.target_pos = target_pos

    # def get_reward(self):
    #     """Uses current pose of sim to return reward."""
    #     x = self.sim.pose[0]
    #     y = self.sim.pose[1]
    #     z = self.sim.pose[2]
    #
    #     vz = self.sim.v[2]
    #
    #     phi = self.sim.pose[3]
    #     theta = self.sim.pose[4]
    #     psi = self.sim.pose[5]
    #
    #     reward = 5 * vz - abs(x) - abs(y) #- abs(phi) - abs(theta) - abs(psi)
    #     return reward

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            pose, step_reward, done, _ = self.sim.step(action) # update the sim pose and velocities
            reward += step_reward
            pose_all.append(pose)
        next_state = np.concatenate(pose_all)
        self.current_steps += self.action_repeat
        if self.current_steps > self.max_steps:
            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.sim.reset()
        state = np.concatenate([state] * self.action_repeat)
        return state
