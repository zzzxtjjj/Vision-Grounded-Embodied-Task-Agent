import gymnasium as gym
import mani_skill.envs


env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict",
    control_mode="pd_joint_delta_pos"
)

obs, info = env.reset(seed=0)

qpos_before = obs["agent"]["qpos"]

action = env.action_space.sample()
action[:] = 0
action[0] = 0.5
action[7] = 1.0

for i in range(10):
    print(f"step number: {i + 1}")
    next_obs, reward, terminated, truncated, info = env.step(action)

    qpos = next_obs["agent"]["qpos"][0]
    joint1_qpos = qpos[0]
    finger1_qpos = qpos[7]
    finger2_qpos = qpos[8]

    print(f"joint1 qpos: {joint1_qpos}")
    print(f"finger1 qpos: {finger1_qpos}")
    print(f"finger2 qpos: {finger2_qpos}")


    qvel = next_obs["agent"]["qvel"][0]
    joint1_qvel = qvel[0]
    
    print(f"joint1 qvel: {joint1_qvel}")

env.close()
