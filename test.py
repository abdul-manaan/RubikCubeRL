import gym
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import rubiks_cube_gym  

# Create and wrap the environment
env = gym.make("rubiks-cube-222-v0")



# Create action noise because TD3 and DDPG use a deterministic policy
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Create RL model
del model # remove to demonstrate saving and loading

model = PPO.load(sys.argv[1])

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    if done == True:
        break

print("End of Algorithm")
