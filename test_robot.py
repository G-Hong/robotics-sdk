import sys
import cv2
sys.path.append('/home/ghong/projects')
import nexodim as nxd


robot = nxd.robots.SO101()
robot.connect(mode="teach")
robot.record(task="pickup", episodes=10)
//
robot.training()
robot.vailidation()
robot.inference()
robot.disconnect()