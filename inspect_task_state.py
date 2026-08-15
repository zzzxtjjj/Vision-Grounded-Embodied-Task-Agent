# 查看 PickCube 中 TCP、方块和目标点的位置关系 Tool Center Point
# 前三个是位置信息position，后四个是空间信息quaternion
import gymnasium as gym
import mani_skill.envs


env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict"
)

obs, info = env.reset(seed=0)

tcp_pose = obs["extra"]["tcp_pose"]
obj_pose = obs["extra"]["obj_pose"]
goal_pose = obs["extra"]["goal_pos"]

tcp_position = tcp_pose[0][0:3]
obj_position = obj_pose[0][0:3]
goal_position = goal_pose[0][0:3]

print(f"TCP pose: {tcp_pose}")
print(f"Object pose: {obj_pose}")
print(f"{goal_pose}")

print(f"TCP position: {tcp_position}")
print(f"Object position: {obj_position}")


# ===========================================================================
calculated_tcp_to_obj = obj_position - tcp_position
print(calculated_tcp_to_obj)
print(obs["extra"]["tcp_to_obj_pos"])
print(calculated_tcp_to_obj == obs["extra"]["tcp_to_obj_pos"])

calculated_obj_to_goal = goal_position - obj_position
print(calculated_obj_to_goal)
print(obs["extra"]["obj_to_goal_pos"])
print(calculated_obj_to_goal == obs["extra"]["obj_to_goal_pos"])