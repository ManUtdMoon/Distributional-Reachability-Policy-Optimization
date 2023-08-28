from typing import Tuple

import gym
import numpy as np
import matplotlib.pyplot as plt

class PointRobot(gym.Env):
    def __init__(self, id=None, seed=None):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.hazard_size = 0.8
        self.hazard_position_list = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]
        self.goal_position = np.array([2.2, 2.2])
        self.goal_size = 0.3
        self.dt = 0.05
        self.state = None
        self.id = id
        self.seed(seed)
        self.last_dist = None
        self.steps = 0

        self._max_episode_steps = 300

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.action_space.seed(seed)
        return [seed]
    
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=[-3.0, -3.0, 0.5, np.pi / 4], high=[3.0, 3.0, 2.0, 3 * np.pi / 4])
        if self.id is not None:
            self.state = np.array([-2.5, -2.5, 2.0, np.pi / 4], dtype=np.float32)
        self.last_dist = np.linalg.norm([self.state[0]-self.goal_position[0], self.state[1]-self.goal_position[1]])
        self.steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        state = self.state + self._dynamics(self.state, action) * self.dt
        reward, done = self.reward_done(state)
        info = self.constraint_value(state)
        self.steps += 1
        if self.steps >= self._max_episode_steps:
            done = True  # Maximum number of steps in an episode reached
        self.state = state
        return self._get_obs(), reward, done, info

    def reward_done(self, state):
        reward = 0.0
        done = False
        dist = np.linalg.norm([state[0]-self.goal_position[0], state[1]-self.goal_position[1]])
        
        reward += (self.last_dist - dist)

        self.last_dist = dist

        if dist <= self.goal_size:
            reward += 1
            done = True
            return reward, done

        done = abs(state[0])>3.0 or abs(state[1])>3.0
        return reward, done
    
    def constraint_value(self, state):
        min_dist = float('inf')
        for hazard_pos in self.hazard_position_list:
            hazard_vec = hazard_pos[:2] - state[:2]
            dist = np.linalg.norm(hazard_vec) - self.hazard_size
            min_dist = min(dist, min_dist)
        
        info = dict(
            cost=(min_dist<=0),
            constraint_value=min_dist,
        )
        return info
    
    @staticmethod
    def _dynamics(s, u):
        v = s[2]
        theta = s[3]

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta], dtype=np.float32)
        return dot_s

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[:3] = self.state[:3]
        theta = self.state[3]
        obs[3] = np.cos(theta)
        obs[4] = np.sin(theta)

        i = 0
        for hazard_pos in self.hazard_position_list:

            hazard_vec = hazard_pos[:2] - self.state[:2]

            dist = np.linalg.norm(hazard_vec)

            velocity_vec = np.array([self.state[3] * np.cos(theta), self.state[3] * np.sin(theta)])
            velocity = np.linalg.norm(velocity_vec)
            velocity = np.clip(velocity, 1e-6, None)
            cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            obs[5+i*3] = dist
            obs[6+i*3] = cos_theta
            obs[7+i*3] = sin_theta

            i += 1

        return obs

    def _get_avoidable(self, state):
        x, y, v, theta = state

        for hazard_position in self.hazard_position_list:
            hazard_vec = hazard_position - np.array([x, y])

            dist = np.linalg.norm(hazard_vec)
            if dist <= self.hazard_size:
                return False


            velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
            velocity = np.linalg.norm(velocity_vec)
            velocity = np.clip(velocity, 1e-6, None)
            cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            delta = self.hazard_size ** 2 - (dist * sin_theta) ** 2
            if cos_theta <= 0 or delta < 0:
                continue

            acc = self.action_space.low[0]
            if np.cross(velocity_vec, hazard_vec) >= 0:
                omega = self.action_space.low[1]
            else:
                omega = self.action_space.high[1]
            action = np.array([acc, omega])
            s = np.copy(state)
            while s[2] > 0:
                s = s + self._dynamics(s, action) * self.dt
                dist = np.linalg.norm([hazard_position[0]-s[0], hazard_position[1]-s[1]])
                if dist <= self.hazard_size:
                    return False
            
        return True

    def plot_map(self, ax, v: float = 2.0, theta: float = np.pi / 4):
        from matplotlib.patches import Circle

        n = 201
        xs = np.linspace(-3.0, 3.0, n)
        ys = np.linspace(-3.0, 3.0, n)
        xs, ys = np.meshgrid(xs, ys)
        vs = v * np.ones_like(xs)
        thetas = theta * np.ones_like(xs)
        obs = np.stack((xs, ys, vs, np.cos(thetas), np.sin(thetas)), axis=-1)

        avoidable = np.zeros_like(xs)
        for i in range(n):
            for j in range(n):
                avoidable[i, j] = float(self._get_avoidable([xs[i, j], ys[i, j], v, theta]))
        ax.contour(xs, ys, avoidable - 0.5, levels=[0], colors='k')
        
        for hazard_position in self.hazard_position_list:
            circle = Circle((hazard_position[0], hazard_position[1]), self.hazard_size, fill=False, linestyle='--', color='k')
            ax.add_patch(circle)
        # Goal
        circle = Circle((self.goal_position[0], self.goal_position[1]), self.goal_size, fill=False, linestyle='--', color='k')
        ax.add_patch(circle)

if __name__ == "__main__":

    env = PointRobot(id=0)

    obs = env.reset()
    print(obs)
    episode_ret, episode_cost, episode_len, i = 0.0, 0.0, 0, 0
    while True:
        # print(f"----------- step {i} -------------")
        # action = env.action_space.sample()
        action = np.array([1,0])
        obs, reward, done, info = env.step(action)
        print(obs)
        cost = info["cost"]
        episode_ret += reward
        episode_len += 1
        episode_cost += cost
        i += 1

        if done:
            break
    
    print('episode_ret', episode_ret)
    print('episode_cost', episode_cost)
    print('episode_len', episode_len)



    


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    env.plot_map(ax)

    plt.show()