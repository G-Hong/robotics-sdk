# NexoDim

A Python framework for standardized robot control.

## Supported Robots
- SO-ARM101 (LeRobot)

## Installation

```bash
cd projects
pip install -e .
```

## Quick Start

```python
import nexodim as nxd

robot = nxd.robots.SO101(id="my_robot")
robot.connect()

obs = robot.get_observation()
print(obs)

robot.disconnect()
```

## API

### connect(mode, use_camera)
Connect to the robot. USB ports are detected automatically.

```python
robot.connect()                          # follower only
robot.connect(mode="teach")              # follower + leader
robot.connect(use_camera=False)          # without camera
```

### get_observation()
Returns joint positions and camera image.

```python
obs = robot.get_observation()
# obs["shoulder_pan.pos"], obs["elbow_flex.pos"], ...
# obs["camera"]  → numpy array (H, W, 3)
```

### send_action(action)
Send joint positions to the robot.

```python
robot.send_action(action)
```

### teleop()
Start teleoperation with the leader arm. Requires `mode="teach"`.

```python
robot.connect(mode="teach")
robot.teleop()  # Ctrl+C to stop
```

### connect_camera(camera_index)
Reconnect or change camera without restarting the robot.

```python
robot.connect_camera()     # retry default camera
robot.connect_camera(1)    # switch to camera 1
```

### disconnect()
Disconnect robot, leader arm, and camera.

```python
robot.disconnect()
```
