import glob
import time
import json
import numpy as np
import os
import cv2
import sys
import select
from threading import Thread
from flask import Flask, Response
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from nexodim.base import NexodimRobot


class SO101(NexodimRobot):

    def __init__(self, id="default", camera_index=None):
        self.id = id
        self.camera_index = camera_index
        self.robot = None
        self.leader = None
        self.camera = None
        self.robot_port = None
        self.leader_port = None
        self.home_position = None

    # ── 내부 유틸 ──

    def _find_port(self, name):
        input(f"[{self.id}] {name} USB를 제거해주세요. 제거 후 엔터")
        before = set(glob.glob("/dev/ttyACM*"))
        input(f"[{self.id}] {name} USB를 다시 꽂아주세요. 꽂은 후 엔터")
        time.sleep(3)
        after = set(glob.glob("/dev/ttyACM*"))
        new_port = list(after - before)[0]
        print(f"[{self.id}] {name} 포트: {new_port}\n")
        return new_port

    def _find_camera(self):
        input(f"[{self.id}] 카메라 USB를 제거해주세요. 제거 후 엔터")
        before = set(glob.glob("/dev/video*"))
        input(f"[{self.id}] 카메라 USB를 다시 꽂아주세요. 꽂은 후 엔터")
        time.sleep(3)
        after = set(glob.glob("/dev/video*"))
        new_devices = sorted(after - before)
        index = int(new_devices[0].replace("/dev/video", ""))
        print(f"[{self.id}] 카메라 인덱스: {index}\n")
        return index

    def _connect_follower(self, calibrate=False):
        if not self.robot_port:
            self.robot_port = self._find_port("Follower Arm")
        config = SOFollowerRobotConfig(
            port=self.robot_port,
            id="my_awesome_follower_arm"
        )
        self.robot = SOFollower(config)
        self.robot.connect(calibrate=calibrate)

    def _connect_leader(self, calibrate=False):
        if not self.leader_port:
            self.leader_port = self._find_port("Leader Arm")
        leader_config = SOLeaderTeleopConfig(
            port=self.leader_port,
            id="my_awesome_leader_arm"
        )
        self.leader = SOLeader(leader_config)
        self.leader.connect(calibrate=calibrate)

    # ── 메인 함수 ──

    def _load_saved_ports(self):
        port_file = os.path.join(os.path.dirname(__file__), "configs", "ports.json")
        if os.path.exists(port_file):
            with open(port_file, "r") as f:
                return json.load(f)
        return {}

    def _save_ports(self):
        port_file = os.path.join(os.path.dirname(__file__), "configs", "ports.json")
        ports = {}
        if self.robot_port:
            ports["follower"] = self.robot_port
        if self.leader_port:
            ports["leader"] = self.leader_port
        if self.camera_index is not None:
            ports["camera"] = self.camera_index
        with open(port_file, "w") as f:
            json.dump(ports, f, indent=2)

    def connect(self, mode="auto", use_camera=True):
        saved = self._load_saved_ports()

        # 팔로워 연결
        if not self.robot_port:
            if "follower" in saved:
                self.robot_port = saved["follower"]
                print(f"[{self.id}] 저장된 포트 사용: {self.robot_port}")
            else:
                self.robot_port = self._find_port("Follower Arm")

        try:
            self._connect_follower(calibrate=False)
        except Exception:
            print(f"[{self.id}] 저장된 포트 실패. 다시 찾기...")
            self.robot_port = self._find_port("Follower Arm")
            self._connect_follower(calibrate=False)

        # 리더 연결
        if mode == "teach":
            if not self.leader_port:
                if "leader" in saved:
                    self.leader_port = saved["leader"]
                    print(f"[{self.id}] 저장된 포트 사용: {self.leader_port}")
                else:
                    self.leader_port = self._find_port("Leader Arm")

            try:
                self._connect_leader(calibrate=False)
            except Exception:
                print(f"[{self.id}] 저장된 포트 실패. 다시 찾기...")
                self.leader_port = self._find_port("Leader Arm")
                self._connect_leader(calibrate=False)

        # 카메라
        if use_camera:
            if self.camera_index is None and "camera" in saved:
                self.camera_index = saved["camera"]
                print(f"[{self.id}] 저장된 카메라 사용: {self.camera_index}")
            self.connect_camera()

            # 연결 실패하면 다시 찾기
            if self.camera is None:
                print(f"[{self.id}] 저장된 카메라 실패. 다시 찾기...")
                self.camera_index = self._find_camera()
                self.connect_camera()

        # 설정 및 홈 포지션 불러오기
        self.setup()
        self._load_home_position()

        # 포트 저장
        self._save_ports()

        print(f"[{self.id}] SO101 연결 완료! (mode={mode})")


    # def connect(self, mode="auto", use_camera=True):
    #     self._connect_follower(calibrate=False)

    #     if mode == "teach":
    #         self._connect_leader(calibrate=False)

    #     if use_camera:
    #         self.connect_camera()

        print(f"[{self.id}] SO101 연결 완료! (mode={mode})")

    def connect_camera(self, camera_index=None):
        if camera_index is not None:
            self.camera_index = camera_index
        elif self.camera_index is None:
            self.camera_index = self._find_camera()

        if self.camera:
            self.camera.release()
        self.camera = cv2.VideoCapture(self.camera_index)
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                preview_path = os.path.join(os.path.expanduser("~"), "projects", "camera_preview.jpg")
                cv2.imwrite(preview_path, frame)
                print(f"[{self.id}] 카메라 연결 완료! 미리보기: {preview_path}")
            else:
                print(f"[{self.id}] 카메라 연결 완료!")
        else:
            print(f"[{self.id}] 카메라 연결 실패.")
            self.camera = None

    def calibrate(self, target="all"):
        if target in ("follower", "all"):
            # 기존 연결 끊고 캘리브레이션 포함 재연결
            try:
                self.robot.disconnect()
            except:
                pass
            self._connect_follower(calibrate=True)
            print(f"[{self.id}] Follower 캘리브레이션 완료!")

        if target in ("leader", "all"):
            try:
                self.leader.disconnect()
            except:
                pass
            self._connect_leader(calibrate=True)
            print(f"[{self.id}] Leader 캘리브레이션 완료!")

    def setup(self):
        config_dir = os.path.join(os.path.dirname(__file__), "configs")

        with open(os.path.join(config_dir, "so101_follower.json"), "r") as f:
            settings = json.load(f)
        for motor_name, params in settings.items():
            for param, value in params.items():
                self.robot.bus.write(param, motor_name, value)
        print(f"[{self.id}] Follower 모터 세팅 적용 완료")

        if self.leader:
            with open(os.path.join(config_dir, "so101_leader.json"), "r") as f:
                settings = json.load(f)
            for motor_name, params in settings.items():
                for param, value in params.items():
                    self.leader.bus.write(param, motor_name, value)
            print(f"[{self.id}] Leader 모터 세팅 적용 완료")

    def first_setup(self):
        # 포트 찾기
        self.robot_port = self._find_port("Follower Arm")
        self.leader_port = self._find_port("Leader Arm")

        # 캘리브레이션 포함 연결
        self._connect_follower(calibrate=True)
        self._connect_leader(calibrate=True)

        # 모터 세팅 적용
        self.setup()

        print(f"[{self.id}] 초기 셋업 완료!")
        self.disconnect()

    # ── 홈 포지션 ──
 
    def _home_position_path(self):
        return os.path.join(os.path.dirname(__file__), "configs", "home_position.json")
 
    def _load_home_position(self):
        """저장된 홈 포지션을 파일에서 로드합니다."""
        path = self._home_position_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                self.home_position = json.load(f)
            print(f"[{self.id}] 저장된 홈 포지션 로드 완료")
        else:
            self.home_position = None
 
    def set_home_position(self, position=None):
        """
        홈 포지션을 설정합니다.
 
        Args:
            position: dict – 관절 위치. None이면 현재 위치를 홈으로 저장.
                예) {"shoulder_pan.pos": 0, "shoulder_lift.pos": -90, ...}
 
        사용법:
            # 방법 1: 현재 위치를 홈으로 저장
            robot.set_home_position()
 
            # 방법 2: 직접 값 지정
            robot.set_home_position({
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": -90.0,
                "elbow_flex.pos": 90.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
            })
        """
        if position is not None:
            self.home_position = position
        else:
            # 현재 관절 위치를 읽어서 홈으로 저장
            obs = self.get_observation()
            joint_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            self.home_position = {}
            for name in joint_names:
                key = f"{name}.pos"
                if key in obs:
                    self.home_position[key] = float(obs[key])
 
        # 파일에 저장
        path = self._home_position_path()
        with open(path, "w") as f:
            json.dump(self.home_position, f, indent=2)
 
        print(f"[{self.id}] 홈 포지션 저장 완료:")
        for k, v in self.home_position.items():
            print(f"    {k}: {v:.2f}")
 
    def go_home(self, duration=3.0, fps=60):
        """
        홈 포지션으로 부드럽게 이동합니다.
 
        Args:
            duration: 이동 시간 (초). 클수록 느리고 안전합니다.
            fps:      보간 주파수
        """
        if self.home_position is None:
            print(f"[{self.id}] 홈 포지션이 설정되지 않았습니다. set_home_position()을 먼저 호출하세요.")
            return
 
        if self.robot is None:
            print(f"[{self.id}] 로봇이 연결되지 않았습니다.")
            return
 
        # 현재 위치 읽기
        obs = self.get_observation()
        joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
 
        current = {}
        for name in joint_names:
            key = f"{name}.pos"
            if key in obs:
                current[key] = float(obs[key])
 
        # 선형 보간으로 부드럽게 이동
        total_steps = int(duration * fps)
        interval = 1.0 / fps
 
        print(f"[{self.id}] 홈 포지션으로 이동 중... ({duration}초)")
 
        for step in range(1, total_steps + 1):
            t = step / total_steps  # 0→1 진행률
 
            # ease-in-out 보간 (더 부드러운 움직임)
            t_smooth = t * t * (3.0 - 2.0 * t)
 
            action = {}
            for key in current:
                if key in self.home_position:
                    start = current[key]
                    end = self.home_position[key]
                    action[key] = start + (end - start) * t_smooth
 
            self.send_action(action)
            time.sleep(interval)
 
        print(f"[{self.id}] 홈 포지션 도착!")        

    # ── 관측/제어 ──

    def get_observation(self):
        obs = self.robot.get_observation()
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                obs["camera"] = frame
        return obs

    def send_action(self, action):
        return self.robot.send_action(action)

    def teleop(self):
        if self.leader is None:
            print("텔레옵 하려면 connect(mode='teach') 로 연결해야 해요!")
            return
        print(f"[{self.id}] 텔레옵 시작! 종료하려면 Ctrl+C")
        try:
            while True:
                action = self.leader.get_action()
                self.robot.send_action(action)
                time.sleep(1/60)
        except KeyboardInterrupt:
            print(f"[{self.id}] 텔레옵 종료")

    def _start_preview_server(self):
        """백그라운드에서 카메라 미리보기 웹서버 실행"""
        app = Flask(__name__)
        robot = self

        @app.route('/')
        def index():
            return '<html><body><h2>NxD Camera Preview</h2><img src="/feed" width="640"></body></html>'

        @app.route('/feed')
        def feed():
            def generate():
                while True:
                    if robot.camera:
                        ret, frame = robot.camera.read()
                        if ret:
                            _, jpeg = cv2.imencode('.jpg', frame)
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(1/30)
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(host='0.0.0.0', port=5000, threaded=True)

    def record(self, task="default_task", episodes=3, fps=30, save_dir=None):
        if self.leader is None:
            print("녹화하려면 connect(mode='teach') 로 연결해야 해요!")
            return
        if self.camera is None:
            print("카메라가 연결되어 있지 않아요! connect(use_camera=True) 확인해주세요.")
            return

        if save_dir is None:
            # 현재 파일(so101.py)의 위치를 기준으로 3단계 위인 Zip0 루트 폴더를 자동으로 찾습니다.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            zip0_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            
            # Zip0/data/so101/task 경로 설정
            save_dir = os.path.join(zip0_root, "data", "so101", task)
        os.makedirs(save_dir, exist_ok=True)

        # 기존 에피소드 이어서
        existing = glob.glob(os.path.join(save_dir, "episode_*"))
        start_ep = len(existing)
        if start_ep > 0:
            print(f"[{self.id}] 기존 {start_ep}개 에피소드 있음. 이어서 녹화!")

        metadata = {"task": task, "episodes": start_ep + episodes, "fps": fps}
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 카메라 미리보기 서버 시작
        if self.camera:
            server = Thread(target=self._start_preview_server, daemon=True)
            server.start()
            print(f"[{self.id}] 카메라 미리보기: http://localhost:5000")

        import select
        import termios

        for ep in range(start_ep, start_ep + episodes):
            ep_dir = os.path.join(save_dir, f"episode_{ep}")
            os.makedirs(ep_dir, exist_ok=True)

            joint_data = []
            video_writer = None
            recording = False

            if self.camera:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    os.path.join(ep_dir, "video.mp4"),
                    fourcc, fps, (640, 480)
                )

            print(f"\n[{self.id}] 에피소드 {ep+1}/{start_ep + episodes}")
            print(f"[{self.id}] 텔레옵 작동 중... 위치 잡으세요")
            print(f"[{self.id}] 엔터: 녹화 시작/종료")

            # 녹화 대기 (텔레옵 유지 및 버퍼 비우기)
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            while True:
                action = self.leader.get_action()
                self.robot.send_action(action)
                time.sleep(1/fps)
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    time.sleep(0.5)  # 중복 입력 방지
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    break

            print(f"[{self.id}] 녹화 시작!")
            recording = True

            # 녹화 중 (텔레옵 + 저장)
            while recording:
                action = self.leader.get_action()
                self.robot.send_action(action)

                obs = self.robot.get_observation()
                
                # 메모리 폭발 방지: 카메라는 영상으로 저장하므로 관절 데이터에서 제외
                joint_only_obs = {k: v for k, v in obs.items() if k != "camera"}
                joint_data.append(joint_only_obs)

                if self.camera and video_writer:
                    ret, frame = self.camera.read()
                    if ret:
                        video_writer.write(frame)

                time.sleep(1/fps)

                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    recording = False

            if video_writer:
                video_writer.release()
            np.save(os.path.join(ep_dir, "joints.npy"), joint_data)
            print(f"[{self.id}] 에피소드 {ep+1} 저장 완료! ({len(joint_data)} 프레임)")

        print(f"\n[{self.id}] 녹화 완료! {save_dir}")

        # 추가 녹화 여부
        choice = input(f"\n[{self.id}] 다른 작업 녹화? (y/n): ").strip().lower()
        if choice == "y":
            new_task = input(f"[{self.id}] 작업 이름: ").strip()
            ep_count = input(f"[{self.id}] 에피소드 수 (기본 3): ").strip()
            ep_count = int(ep_count) if ep_count else 3
            self.record(task=new_task, episodes=ep_count, fps=fps)

    # ── 해제 ──

    def disconnect_leader(self):
        if self.leader:
            try:
                self.leader.disconnect()
            except Exception as e:
                print(f"[{self.id}] Leader 해제 중 경고: {e}")
            self.leader = None
            print(f"[{self.id}] Leader Arm 해제 완료")

    def disconnect_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None
            print(f"[{self.id}] Camera 해제 완료")

    def disconnect(self):
        if self.robot:
            try:
                self.robot.disconnect()
            except Exception as e:
                print(f"[{self.id}] Main Robot 해제 중 경고: {e}")
            print(f"[{self.id}] Main Robot 해제 완료")
        self.disconnect_leader()
        self.disconnect_camera()

    def safe_disconnect(self, duration=3.0):
        """홈 포지션으로 이동 후 안전하게 연결 해제합니다."""
        if self.home_position is not None and self.robot is not None:
            print(f"[{self.id}] 안전 종료: 홈 포지션으로 이동 중...")
            self.go_home(duration=duration)
        self.disconnect()