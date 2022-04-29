# from tkinter.tix import Tree
import gym  
import rubiks_cube_gym  
import numpy as np
import time
from stable_baselines3 import PPO

class RandomAgent:
    def predict(self, obs):
        return np.random.randint(3), None

# print(gym.envs.registry.all())

# if __name__ == "__main__":
#     env = gym.make("rubiks-cube-222-v0")

#     env.scramble_params = 20  # number of random actions to do for scrambling (called in reset), =0 don't scramble the cube

#     env.screen_width = 600  # you can reduce the screen size if FPS are too low
#     env.screen_height = 600

#     env.cap_fps = 10  # env assumes that you are close to this fps (controls might be weird if it's too far away from it)
#     env.rotation_step = 5  # higher values makes the animation of rotating take less frames and vice versa, =90 rotations aren't animated
#     env.max_steps = 50  # number of steps after the environment will return done = False, default None

#     model = RandomAgent()

#     obs = env.reset(scramble=False)
#     done = False
#     print(env.observation_space)
#     while not done:
#         print(env.render(mode="rgb_array"))
#         action, _ = model.predict(obs)

#         obs, reward, done, _ = env.step(action)
#         print(action, reward)
#         time.sleep(10)



#     env.close()



env = gym.make("rubiks-cube-222-v0")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

for i in range(10):
    obs = env.reset()
    env.render(mode="human")
    time.sleep(10)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    time.sleep(10)
    if done:
        obs = env.reset()
env.close()