"""
SmolVLA Policy – NexoDim 표준 규격 구현

LeRobot의 SmolVLA (450M VLA 모델)를 NexodimPolicies 인터페이스로 감싸서
로드 → 학습 → 검증 → 추론 → 저장을 단순한 함수 호출로 사용할 수 있게 합니다.

사용 예시:
    import nexodim as nxd

    policy = nxd.policies.vla.SmolVLA()
    policy.load_policy("lerobot/smolvla_base")
    policy.train_policy(dataset_repo_id="user/my_dataset", steps=20000)
    policy.save_policy("outputs/my_model")

    # 로봇 연결 후 추론
    robot = nxd.robots.SO101()
    robot.connect()
    obs = robot.get_observation()
    action = policy.inference_policy(obs)
    robot.send_action(action)
"""

import os
import time
import json
from pathlib import Path

import torch
import numpy as np

from nexodim.base import NexodimPolicies


class SmolVLA(NexodimPolicies):
    """
    SmolVLA VLA 정책 래퍼.

    LeRobot의 SmolVLAPolicy를 NexoDim 표준(NexodimPolicies)에 맞춰 감싼 클래스.
    복잡한 설정 없이 load → train → validate → inference → save 흐름을 제공합니다.
    """

    def __init__(self, device=None):
        """
        Args:
            device: 사용할 디바이스. None이면 자동 감지 (cuda > mps > cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.preprocessor = None
        self.postprocessor = None
        self.config = None
        self.model_id = None

        # 학습 관련 상태
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None
        self.dataloader = None
        self.dataset_metadata = None
        self.dataset_features = None

        # 추론 관련 상태
        self.task = ""
        self.robot_type = ""

        # 학습 이력
        self.train_history = []

        print(f"[SmolVLA] 디바이스: {self.device}")

    # ══════════════════════════════════════════════════════
    #  load_policy – 모델 로드
    # ══════════════════════════════════════════════════════

    def load_policy(self, model_path="lerobot/smolvla_base", task="", robot_type=""):
        """
        SmolVLA 모델을 로드합니다.

        Args:
            model_path: HuggingFace Hub repo_id 또는 로컬 체크포인트 경로
                        예) "lerobot/smolvla_base", "outputs/train/my_model/checkpoints/last/pretrained_model"
            task:       추론 시 사용할 자연어 작업 명령 (예: "pick up the red block")
            robot_type: 로봇 타입 (예: "so101_follower"). 멀티 로봇 데이터셋 사용 시 필요.
        """
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.model_id = model_path
        self.task = task
        self.robot_type = robot_type

        print(f"[SmolVLA] 모델 로딩 중: {model_path}")
        self.model = SmolVLAPolicy.from_pretrained(model_path)
        self.config = self.model.config

        # pre/post 프로세서 생성
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self.model.to(self.device)
        print(f"[SmolVLA] 모델 로드 완료! (device={self.device})")
        return self

    # ══════════════════════════════════════════════════════
    #  train_policy – 학습
    # ══════════════════════════════════════════════════════

    def train_policy(
        self,
        dataset_repo_id,
        steps=20000,
        batch_size=64,
        lr=1e-4,
        warmup_steps=1000,
        save_freq=5000,
        log_freq=100,
        output_dir=None,
        use_amp=False,
        num_workers=4,
        resume_from=None,
        gradient_clip_max_norm=1.0,
    ):
        """
        SmolVLA 모델을 파인튜닝합니다.

        Args:
            dataset_repo_id: HuggingFace Hub 데이터셋 ID (예: "user/my_dataset")
            steps:           총 학습 스텝 수
            batch_size:      배치 크기 (VRAM에 맞게 조절. 6GB→16, 12GB→44, 24GB+→64)
            lr:              학습률
            warmup_steps:    워밍업 스텝 수
            save_freq:       체크포인트 저장 빈도 (스텝 단위)
            log_freq:        로그 출력 빈도 (스텝 단위)
            output_dir:      출력 디렉토리. None이면 자동 생성.
            use_amp:         Mixed Precision (AMP) 사용 여부. VRAM 부족 시 True 권장.
            num_workers:     DataLoader worker 수
            resume_from:     이어서 학습할 체크포인트 경로
            gradient_clip_max_norm: 그래디언트 클리핑 최대 노름
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.feature_utils import dataset_to_policy_features
        from lerobot.configs.types import FeatureType
        from lerobot.policies.factory import make_pre_post_processors

        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_policy()를 먼저 호출하세요."
            )

        if output_dir is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = os.path.join("outputs", "train", f"smolvla_{timestamp}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SmolVLA] 데이터셋 로딩: {dataset_repo_id}")
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)

        # 데이터셋 피처를 정책 피처로 변환
        features = dataset_to_policy_features(self.dataset_metadata.features)
        output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
        input_features = {
            key: ft for key, ft in features.items() if key not in output_features
        }

        # 프리프로세서 업데이트 (데이터셋 통계 기반 정규화)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            self.model_id,
            dataset_stats=self.dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # delta_timestamps 설정
        fps = self.dataset_metadata.fps
        delta_timestamps = self._build_delta_timestamps(fps)

        # 데이터셋 & 데이터로더 생성
        self.dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type != "cpu",
            drop_last=True,
        )

        # 옵티마이저 & 스케줄러
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        if warmup_steps > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.lr_scheduler = None

        # AMP scaler
        scaler = torch.amp.GradScaler(self.device.type) if use_amp else None

        # 학습 시작
        self.model.train()
        self.train_history = []
        start_step = 0

        if resume_from:
            start_step = self._load_training_state(resume_from)
            print(f"[SmolVLA] 체크포인트에서 이어서 학습: step {start_step}")

        print(f"[SmolVLA] 학습 시작!")
        print(f"  - 데이터셋: {dataset_repo_id}")
        print(f"  - 총 스텝: {steps}")
        print(f"  - 배치 크기: {batch_size}")
        print(f"  - 학습률: {lr}")
        print(f"  - 출력: {output_dir}")
        print(f"  - AMP: {use_amp}")
        print()

        step = start_step
        done = False
        epoch = 0

        while not done:
            epoch += 1
            for batch in self.dataloader:
                # 전처리
                batch = self.preprocessor(batch)

                # Forward + Backward
                self.optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        loss, output_dict = self.model.forward(batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip_max_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss, output_dict = self.model.forward(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), gradient_clip_max_norm
                    )
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 로깅
                loss_val = loss.item()
                self.train_history.append({"step": step, "loss": loss_val, "epoch": epoch})

                if step % log_freq == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  step: {step}/{steps}  |  loss: {loss_val:.4f}  |  "
                        f"lr: {current_lr:.2e}  |  epoch: {epoch}"
                    )

                # 체크포인트 저장
                if step > 0 and step % save_freq == 0:
                    ckpt_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                    self._save_checkpoint(ckpt_dir, step)
                    print(f"  [checkpoint] step {step} → {ckpt_dir}")

                step += 1
                if step >= steps:
                    done = True
                    break

        # 최종 체크포인트 저장
        last_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
        self._save_checkpoint(last_dir, step)

        # 학습 이력 저장
        history_path = output_dir / "train_history.json"
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=2)

        print(f"\n[SmolVLA] 학습 완료! ({step} 스텝)")
        print(f"  - 최종 체크포인트: {last_dir}")
        print(f"  - 학습 이력: {history_path}")
        print(f"  - 최종 loss: {self.train_history[-1]['loss']:.4f}")

        return self.train_history

    # ══════════════════════════════════════════════════════
    #  validate_policy – 검증
    # ══════════════════════════════════════════════════════

    def validate_policy(
        self,
        dataset_repo_id=None,
        num_batches=50,
        batch_size=32,
    ):
        """
        모델의 validation loss를 계산합니다.

        Args:
            dataset_repo_id: 검증용 데이터셋 (None이면 학습에 사용한 데이터셋 재사용)
            num_batches:     검증할 배치 수
            batch_size:      배치 크기

        Returns:
            dict: {"avg_loss": float, "num_batches": int, "losses": list}
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_pre_post_processors

        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_policy()를 먼저 호출하세요."
            )

        # 데이터셋 준비
        if dataset_repo_id is not None:
            metadata = LeRobotDatasetMetadata(dataset_repo_id)
            fps = metadata.fps
            delta_timestamps = self._build_delta_timestamps(fps)
            val_dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)

            preprocessor, _ = make_pre_post_processors(
                self.config,
                self.model_id,
                dataset_stats=metadata.stats,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        elif self.dataset is not None:
            val_dataset = self.dataset
            preprocessor = self.preprocessor
        else:
            raise RuntimeError(
                "dataset_repo_id를 지정하거나 train_policy()를 먼저 실행하세요."
            )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.device.type != "cpu",
            drop_last=True,
        )

        # 검증 루프
        self.model.eval()
        losses = []

        print(f"[SmolVLA] 검증 시작 ({num_batches} 배치)")
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                batch = preprocessor(batch)
                loss, _ = self.model.forward(batch)
                losses.append(loss.item())

        self.model.train()

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        result = {
            "avg_loss": avg_loss,
            "num_batches": len(losses),
            "losses": losses,
        }

        print(f"[SmolVLA] 검증 완료!")
        print(f"  - 평균 loss: {avg_loss:.4f}")
        print(f"  - 검증 배치 수: {len(losses)}")

        return result

    # ══════════════════════════════════════════════════════
    #  inference_policy – 추론 (로봇 제어)
    # ══════════════════════════════════════════════════════

    def inference_policy(self, observation, task=None, robot_type=None):
        """
        관측(observation)으로부터 로봇 액션을 추론합니다.

        NexodimRobot.get_observation()의 반환값을 그대로 넣으면 됩니다.

        Args:
            observation: dict – 로봇의 현재 관측
                필수 키:
                  - "observation.state" 또는 각 관절 이름 (shoulder_pan.pos 등)
                  - "observation.images.XXX" 또는 "camera" (numpy 이미지)
            task:       자연어 작업 명령. None이면 load_policy 시 설정한 값 사용.
            robot_type: 로봇 타입. None이면 load_policy 시 설정한 값 사용.

        Returns:
            dict: 로봇에 보낼 액션 (send_action에 직접 전달 가능)
        """
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.datasets.feature_utils import hw_to_dataset_features

        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_policy()를 먼저 호출하세요."
            )

        task = task if task is not None else self.task
        robot_type = robot_type if robot_type is not None else self.robot_type

        self.model.eval()

        # NexodimRobot의 관측을 LeRobot 형식으로 변환
        obs = self._convert_observation(observation)

        # dataset_features가 없으면 모델의 config에서 추출
        if self.dataset_features is None:
            self.dataset_features = self._build_dataset_features_from_config()

        # 추론 프레임 구성
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=self.dataset_features,
            device=self.device,
            task=task,
            robot_type=robot_type,
        )

        # 전처리 → 추론 → 후처리
        obs_frame = self.preprocessor(obs_frame)

        with torch.no_grad():
            action = self.model.select_action(obs_frame)

        action = self.postprocessor(action)
        action = make_robot_action(action, self.dataset_features)

        return action

    def run_inference_loop(
        self,
        robot,
        task=None,
        max_steps=200,
        fps=30,
    ):
        """
        로봇과 연결해서 연속 추론 루프를 실행합니다.

        Args:
            robot:     NexodimRobot 인스턴스 (이미 connect 된 상태)
            task:      자연어 작업 명령
            max_steps: 최대 스텝 수
            fps:       제어 주기

        사용 예시:
            robot = nxd.robots.SO101()
            robot.connect()
            policy = nxd.policies.vla.SmolVLA()
            policy.load_policy("user/my_finetuned_model", task="pick up the block")
            policy.run_inference_loop(robot, max_steps=300)
        """
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_policy()를 먼저 호출하세요."
            )

        task = task if task is not None else self.task
        interval = 1.0 / fps

        print(f"[SmolVLA] 추론 루프 시작! (task='{task}', max_steps={max_steps})")
        print(f"[SmolVLA] 종료하려면 Ctrl+C")

        self.model.eval()

        try:
            for step in range(max_steps):
                t_start = time.time()

                obs = robot.get_observation()
                action = self.inference_policy(obs, task=task)
                robot.send_action(action)

                elapsed = time.time() - t_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step % 30 == 0:
                    actual_fps = 1.0 / max(elapsed, 1e-6)
                    print(f"  step {step}/{max_steps}  |  fps: {actual_fps:.1f}")

        except KeyboardInterrupt:
            print(f"\n[SmolVLA] 추론 루프 종료 (사용자 중단, step={step})")
            return step

        print(f"[SmolVLA] 추론 루프 완료! ({max_steps} 스텝)")
        return max_steps

    # ══════════════════════════════════════════════════════
    #  save_policy – 저장
    # ══════════════════════════════════════════════════════

    def save_policy(self, save_path=None, push_to_hub=False, hub_repo_id=None):
        """
        모델을 저장합니다.

        Args:
            save_path:    로컬 저장 경로. None이면 자동 생성.
            push_to_hub:  HuggingFace Hub에 업로드할지 여부
            hub_repo_id:  Hub repo ID (예: "user/my_smolvla_model")
        """
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_policy()를 먼저 호출하세요."
            )

        if save_path is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join("outputs", "saved_models", f"smolvla_{timestamp}")
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 모델 & 프로세서 저장
        self.model.save_pretrained(save_path)
        if self.preprocessor is not None:
            self.preprocessor.save_pretrained(save_path)
        if self.postprocessor is not None:
            self.postprocessor.save_pretrained(save_path)

        print(f"[SmolVLA] 모델 저장 완료: {save_path}")

        # Hub 업로드
        if push_to_hub:
            if hub_repo_id is None:
                raise ValueError("push_to_hub=True 시 hub_repo_id를 지정해야 합니다.")
            self.model.push_to_hub(hub_repo_id)
            if self.preprocessor is not None:
                self.preprocessor.push_to_hub(hub_repo_id)
            if self.postprocessor is not None:
                self.postprocessor.push_to_hub(hub_repo_id)
            print(f"[SmolVLA] Hub 업로드 완료: {hub_repo_id}")

        return save_path

    # ══════════════════════════════════════════════════════
    #  내부 유틸리티 메서드
    # ══════════════════════════════════════════════════════

    def _build_delta_timestamps(self, fps):
        """모델 config의 delta indices를 사용하여 delta_timestamps를 구성합니다."""
        cfg = self.config

        def _indices_to_timestamps(indices, fps):
            if indices is None:
                return [0.0]
            return [i / fps for i in indices]

        obs_indices = getattr(cfg, "observation_delta_indices", None)
        act_indices = getattr(cfg, "action_delta_indices", None)

        delta_timestamps = {
            "observation.state": _indices_to_timestamps(obs_indices, fps),
            "action": _indices_to_timestamps(act_indices, fps),
        }

        # 이미지 피처 추가
        image_features = getattr(cfg, "image_features", {})
        for key in image_features:
            delta_timestamps[key] = _indices_to_timestamps(obs_indices, fps)

        return delta_timestamps

    def _convert_observation(self, obs):
        """
        NexodimRobot.get_observation() 형식을 LeRobot 추론 형식으로 변환합니다.

        SO101의 get_observation()은 다음과 같은 형태를 반환:
          - "shoulder_pan.pos", "shoulder_lift.pos", ... 등 관절 값
          - "camera" → numpy 이미지 (H, W, 3) BGR

        이를 LeRobot이 기대하는 형식으로 변환:
          - "observation.state" → tensor
          - "observation.images.front" → tensor (C, H, W) RGB
        """
        import cv2

        converted = {}

        # 관절 상태 수집
        joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        state_values = []

        # 이미 observation.state 형태로 들어오는 경우
        if "observation.state" in obs:
            converted["observation.state"] = obs["observation.state"]
        else:
            # NexodimRobot 형식의 개별 관절 값들을 모아서 state 벡터 생성
            for name in joint_names:
                key_pos = f"{name}.pos"
                if key_pos in obs:
                    val = obs[key_pos]
                    if isinstance(val, torch.Tensor):
                        state_values.append(val.item() if val.dim() == 0 else val)
                    else:
                        state_values.append(float(val))

            if state_values:
                converted["observation.state"] = torch.tensor(
                    state_values, dtype=torch.float32
                )

        # 카메라 이미지 변환
        # "camera" 키 → "observation.images.front"
        if "camera" in obs and obs["camera"] is not None:
            frame = obs["camera"]
            if isinstance(frame, np.ndarray):
                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # (H, W, C) → (C, H, W), 0~255 → 0~1 float
                frame_tensor = (
                    torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                )
                converted["observation.images.front"] = frame_tensor
            elif isinstance(frame, torch.Tensor):
                converted["observation.images.front"] = frame

        # 이미 LeRobot 형식의 이미지 키가 있으면 그대로 전달
        for key, val in obs.items():
            if key.startswith("observation.images."):
                converted[key] = val

        # 나머지 observation. 키도 전달
        for key, val in obs.items():
            if key.startswith("observation.") and key not in converted:
                converted[key] = val

        return converted

    def _build_dataset_features_from_config(self):
        """모델 config에서 dataset_features를 구성합니다."""
        from lerobot.configs.types import FeatureType, PolicyFeature

        features = {}

        # input_features
        input_features = getattr(self.config, "input_features", {})
        for key, feat in input_features.items():
            features[key] = feat

        # output_features
        output_features = getattr(self.config, "output_features", {})
        for key, feat in output_features.items():
            features[key] = feat

        return features

    def _save_checkpoint(self, ckpt_dir, step):
        """체크포인트를 저장합니다."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        if self.preprocessor is not None:
            self.preprocessor.save_pretrained(ckpt_dir)
        if self.postprocessor is not None:
            self.postprocessor.save_pretrained(ckpt_dir)

        # 학습 상태 저장
        training_state_dir = ckpt_dir.parent / "training_state"
        training_state_dir.mkdir(parents=True, exist_ok=True)

        # 옵티마이저 상태
        torch.save(self.optimizer.state_dict(), training_state_dir / "optimizer.pt")

        # 스케줄러 상태
        if self.lr_scheduler is not None:
            torch.save(
                self.lr_scheduler.state_dict(), training_state_dir / "scheduler.pt"
            )

        # 학습 스텝
        with open(training_state_dir / "training_step.json", "w") as f:
            json.dump({"step": step}, f)

    def _load_training_state(self, ckpt_path):
        """체크포인트에서 학습 상태를 복원합니다."""
        ckpt_path = Path(ckpt_path)

        training_state_dir = ckpt_path / "training_state"

        # 옵티마이저 복원
        opt_path = training_state_dir / "optimizer.pt"
        if opt_path.exists() and self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(opt_path, weights_only=True))

        # 스케줄러 복원
        sched_path = training_state_dir / "scheduler.pt"
        if sched_path.exists() and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(sched_path, weights_only=True))

        # 학습 스텝 복원
        step_path = training_state_dir / "training_step.json"
        if step_path.exists():
            with open(step_path, "r") as f:
                data = json.load(f)
            return data.get("step", 0)

        return 0

    def set_dataset_features(self, robot):
        """
        로봇의 피처 정보를 사용하여 dataset_features를 설정합니다.
        추론 전에 호출하면 관측 키 매핑이 정확해집니다.

        Args:
            robot: NexodimRobot 인스턴스 (connect 된 상태, 내부에 lerobot 로봇 객체 보유)
        """
        from lerobot.datasets.feature_utils import hw_to_dataset_features

        lerobot_robot = getattr(robot, "robot", None)
        if lerobot_robot is None:
            print("[SmolVLA] 경고: 로봇에서 lerobot 객체를 찾을 수 없습니다.")
            return

        action_features = hw_to_dataset_features(lerobot_robot.action_features, "action")
        obs_features = hw_to_dataset_features(
            lerobot_robot.observation_features, "observation"
        )
        self.dataset_features = {**action_features, **obs_features}
        print(f"[SmolVLA] 데이터셋 피처 설정 완료 ({len(self.dataset_features)} 피처)")