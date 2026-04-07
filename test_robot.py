import sys
import cv2
sys.path.append('/home/ghong/projects')
import nexodim as nxd


robot = nxd.robots.SO101()
policy = nxd.policies.vla.SmolVLA()

robot.connect()
policy.load_policy(
    task="pick up the cup"
)
policy.set_dataset_features(robot)
obs = robot.get_observation()
action = policy.inference_policy(obs)
robot.send_action(action)
print(f'Action : {action}')

robot.disconnect()

"""
robot.connect(mode="teach")
robot.record(task='pick up', episodes=3)
robot.disconnect()
"""
