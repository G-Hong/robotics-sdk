import os
import glob
import json
import numpy as np
import cv2
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert_data():
    # 1. 경로 자동 설정 (Zip0 기준)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 아까 녹화했던 원본 데이터 경로
    raw_dir = os.path.join(current_dir, "data", "so101", "pick up the cup")
    
    # 3단계 학습을 위해 새로 만들어질 LeRobot 표준 데이터셋 경로
    out_dir = os.path.join(current_dir, "data", "lerobot_dataset", "pick_up_the_cup")
    repo_id = "local/pick_up_the_cup"

    print(f"원본 데이터 경로: {raw_dir}")
    if not os.path.exists(raw_dir):
        print("원본 데이터 폴더를 찾을 수 없습니다! 경로를 확인하세요.")
        return

    with open(os.path.join(raw_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    fps = meta["fps"]

    # 2. LeRobot 표준 데이터셋 규격(Schema) 정의
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
            "info": {"video.fps": fps, "video.codec": "mp4v", "video.pix_fmt": "yuv420p"}
        }
    }

    # 3. LeRobotDataset 생성 (v3.0 방식 적용)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_dir,
        use_videos=True,
    )

    ep_dirs = sorted(glob.glob(os.path.join(raw_dir, "episode_*")))
    for ep_idx, ep_dir in enumerate(ep_dirs):
        print(f"[{ep_idx+1}/{len(ep_dirs)}] {os.path.basename(ep_dir)} 변환 중...")
        
        joints = np.load(os.path.join(ep_dir, "joints.npy"), allow_pickle=True)
        cap = cv2.VideoCapture(os.path.join(ep_dir, "video.mp4"))
        
        for joint_obs in joints:
            ret, frame = cap.read()
            if not ret:
                break
                
            # OpenCV(BGR)를 PyTorch 표준(RGB, C x H x W)으로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
            
            # 관절(State) 값 추출
            joint_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                           "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
            state_arr = [joint_obs[k] for k in joint_names]
            state_tensor = torch.tensor(state_arr, dtype=torch.float32)
            
            # Action 할당 (텔레옵이므로 현재 상태를 목표 행동으로 취급)
            action_tensor = state_tensor.clone()
            
            # dataset.add_frame()을 사용하여 1프레임씩 추가
            dataset.add_frame({
                "observation.state": state_tensor,
                "action": action_tensor,
                "observation.images.camera1": frame_tensor,
                "task": meta["task"], # 프레임마다 태스크 정보를 추가해야 합니다.
            })
        
        cap.release()
        dataset.save_episode() # 에피소드 단위로 저장
        
    dataset.finalize() # 데이터셋 작성 완료 (v3.0 필수 과정)
    print(f"\n✅ 변환 완료! 새로운 데이터셋 폴더가 생성되었습니다:")
    print(out_dir)

if __name__ == "__main__":
    convert_data()