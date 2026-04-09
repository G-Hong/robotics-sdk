"""
NexoDim 전체 파이프라인 테스트

1. 홈 포지션 설정
2. 에피소드 녹화 (텔레옵)
3. SmolVLA 파인튜닝
4. 추론 (로봇이 실제로 작업 수행)
5. 홈 포지션으로 복귀 후 안전 종료

사용법:
    python test_full_pipeline.py

각 단계를 개별 실행하려면 맨 아래 main()에서 원하는 함수만 호출하세요.
"""

import sys
import time
sys.path.append('/home/nyoung/dev/NxD/Zip0')
import nexodim as nxd


# ══════════════════════════════════════════════════════
#  1단계: 홈 포지션 설정
# ══════════════════════════════════════════════════════

def step1_set_home():
    """로봇을 원하는 시작 자세로 놓고 홈 포지션으로 저장합니다."""
    robot = nxd.robots.SO101()
    robot.connect(mode="teach")

    print("\n" + "=" * 50)
    print("  1단계: 홈 포지션 설정")
    print("=" * 50)
    print("  리더 암으로 팔로워를 안전한 시작 위치로 옮기세요.")
    print("  이 위치가 매 에피소드의 시작/끝 위치가 됩니다.")
    
    print("\n  위치를 잡았으면 엔터를 누르세요...")
    import select
    while True:
        action = robot.leader.get_action()
        robot.robot.send_action(action)
        time.sleep(1/30)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break

    robot.set_home_position()

    print("\n  홈 포지션 테스트: 팔을 다른 곳으로 옮긴 뒤 엔터를 누르세요.")
    print("  엔터를 누르면 홈으로 돌아갑니다...")
    while True:
        action = robot.leader.get_action()
        robot.robot.send_action(action)
        time.sleep(1/30)
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break

    robot.go_home(duration=3.0)

    print("\n  홈 포지션이 올바른가요?")
    choice = input("  (y: 저장 완료 / n: 다시 설정): ").strip().lower()
    if choice == "n":
        robot.disconnect()
        return step1_set_home()

    robot.disconnect()
    print("\n  1단계 완료!\n")


# ══════════════════════════════════════════════════════
#  2단계: 에피소드 녹화
# ══════════════════════════════════════════════════════

def step2_record(task="pick up the cup", episodes=10):
    """텔레옵으로 작업 데이터를 녹화합니다."""
    robot = nxd.robots.SO101()
    robot.connect(mode="teach")

    print("\n" + "=" * 50)
    print("  2단계: 데이터 녹화")
    print(f"  작업: '{task}'")
    print(f"  에피소드 수: {episodes}")
    print("=" * 50)

    # 녹화 전 홈 포지션으로 이동
    if robot.home_position is not None:
        print("\n  홈 포지션으로 이동 중...")
        robot.go_home(duration=2.0)

    input("\n  준비되면 엔터를 누르세요...")

    robot.record(task=task, episodes=episodes, fps=30)

    # 녹화 끝나고 홈으로
    if robot.home_position is not None:
        robot.go_home(duration=2.0)

    robot.disconnect()
    print("\n  2단계 완료!\n")


# ══════════════════════════════════════════════════════
#  3단계: 학습
# ══════════════════════════════════════════════════════

def step3_train(dataset_repo_id = '/home/nyoung/dev/NxD/Zip0/data/lerobot_dataset/pick_up_the_cup', steps=20000, batch_size=64):
    """SmolVLA 모델을 파인튜닝합니다."""
    print("\n" + "=" * 50)
    print("  3단계: SmolVLA 파인튜닝")
    print(f"  데이터셋: {dataset_repo_id}")
    print(f"  스텝: {steps}")
    print(f"  배치: {batch_size}")
    print("=" * 50)

    policy = nxd.policies.vla.SmolVLA()
    policy.load_policy("lerobot/smolvla_base")

    history = policy.train_policy(
        dataset_repo_id=dataset_repo_id,
        steps=steps,
        batch_size=batch_size,
        lr=1e-4,
        warmup_steps=1000,
        save_freq=5000,
        log_freq=100,
        output_dir="outputs/train/my_smolvla",
        use_amp=False,
    )

    # 검증
    val_result = policy.validate_policy(num_batches=30)

    # 저장
    policy.save_policy("outputs/saved_models/my_smolvla_finetuned")

    print(f"\n  3단계 완료!")
    print(f"  최종 loss: {history[-1]['loss']:.4f}")
    print(f"  검증 loss: {val_result['avg_loss']:.4f}\n")

    return policy


# ══════════════════════════════════════════════════════
#  4단계: 추론 (실제 로봇 제어)
# ══════════════════════════════════════════════════════

def step4_inference(
    model_path="outputs/saved_models/my_smolvla_finetuned",
    task="pick up the cup",
    max_steps=300,
    fps=30,
):
    """학습된 모델로 로봇을 제어합니다."""
    robot = nxd.robots.SO101()
    robot.connect()

    policy = nxd.policies.vla.SmolVLA()
    policy.load_policy(model_path, task=task)

    print("\n" + "=" * 50)
    print("  4단계: 추론 (로봇 제어)")
    print(f"  모델: {model_path}")
    print(f"  작업: '{task}'")
    print(f"  최대 스텝: {max_steps}")
    print("=" * 50)

    # 홈 포지션에서 시작
    if robot.home_position is not None:
        print("\n  홈 포지션으로 이동 중...")
        robot.go_home(duration=2.0)

    print("\n  로봇이 움직이기 시작합니다!")
    print("  Ctrl+C 로 언제든 정지 가능")
    input("\n  준비되면 엔터를 누르세요...")

    # 추론 루프
    interval = 1.0 / fps
    step = 0

    try:
        for step in range(max_steps):
            t_start = time.time()

            obs = robot.get_observation()
            action = policy.inference_policy(obs)
            robot.send_action(action)

            if step % 10 == 0:
                elapsed = time.time() - t_start
                actual_fps = 1.0 / max(elapsed, 1e-6)
                gripper = action.get("gripper.pos", 0)
                print(
                    f"  step {step:4d}/{max_steps}  |  "
                    f"fps: {actual_fps:5.1f}  |  "
                    f"gripper: {gripper:.2f}"
                )

            elapsed = time.time() - t_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n  사용자 중단! (step={step})")

    # 안전 종료: 홈으로 돌아간 뒤 연결 해제
    print("\n  홈 포지션으로 복귀 후 종료합니다...")
    robot.safe_disconnect(duration=3.0)
    print("\n  4단계 완료!\n")


# ══════════════════════════════════════════════════════
#  빠른 테스트: 베이스 모델로 동작 확인
# ══════════════════════════════════════════════════════

def quick_test(task="pick up the cup", max_steps=100):
    """
    파인튜닝 없이 베이스 모델로 로봇 동작을 빠르게 테스트합니다.
    로봇이 의미 있는 작업을 수행하지는 않지만,
    모델→로봇 파이프라인이 정상 동작하는지 확인할 수 있습니다.
    """
    robot = nxd.robots.SO101()
    robot.connect()

    # 홈 포지션이 없으면 현재 위치를 홈으로 설정
    if robot.home_position is None:
        print("\n  홈 포지션이 없습니다. 현재 위치를 홈으로 저장합니다.")
        robot.set_home_position()

    policy = nxd.policies.vla.SmolVLA()
    policy.load_policy("lerobot/smolvla_base", task=task)

    print("\n" + "=" * 50)
    print("  빠른 테스트 (베이스 모델)")
    print(f"  작업: '{task}'")
    print(f"  스텝: {max_steps}")
    print("=" * 50)
    print("\n  주의: 베이스 모델은 파인튜닝 전이라")
    print("  정확한 작업 수행은 어렵습니다.")
    print("  Ctrl+C 로 언제든 정지 가능")
    input("\n  준비되면 엔터를 누르세요...")

    interval = 1.0 / 30
    step = 0

    try:
        for step in range(max_steps):
            t_start = time.time()

            obs = robot.get_observation()
            action = policy.inference_policy(obs)
            robot.send_action(action)

            if step % 10 == 0:
                elapsed = time.time() - t_start
                actual_fps = 1.0 / max(elapsed, 1e-6)
                print(f"  step {step:4d}/{max_steps}  |  fps: {actual_fps:5.1f}")

            elapsed = time.time() - t_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n  사용자 중단! (step={step})")

    # 안전 종료
    print("\n  홈 포지션으로 복귀 후 종료합니다...")
    robot.safe_disconnect(duration=3.0)
    print("\n  테스트 완료!\n")


# ══════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 50)
    print("  NexoDim 전체 파이프라인")
    print("=" * 50)
    print("  1. 홈 포지션 설정")
    print("  2. 에피소드 녹화 (10개)")
    print("  3. SmolVLA 파인튜닝")
    print("  4. 추론 테스트")
    print("  5. 빠른 테스트 (베이스 모델)")
    print("  q. 종료")
    print()

    choice = input("  실행할 단계를 선택하세요 (1-5, q): ").strip()

    if choice == "1":
        step1_set_home()

    elif choice == "2":
        task = input("  작업 이름 (기본: pick up the cup): ").strip()
        task = task if task else "pick up the cup"
        ep = input("  에피소드 수 (기본: 10): ").strip()
        ep = int(ep) if ep else 10
        step2_record(task=task, episodes=ep)

    elif choice == "3":
        ds = input("  데이터셋 repo_id: ").strip()
        if not ds:
            print("  데이터셋을 입력해주세요!")
            return
        steps = input("  학습 스텝 수 (기본: 20000): ").strip()
        steps = int(steps) if steps else 20000
        bs = input("  배치 크기 (기본: 64): ").strip()
        bs = int(bs) if bs else 64
        step3_train(dataset_repo_id=ds, steps=steps, batch_size=bs)

    elif choice == "4":
        model = input("  모델 경로 (기본: outputs/saved_models/my_smolvla_finetuned): ").strip()
        model = model if model else "outputs/saved_models/my_smolvla_finetuned"
        task = input("  작업 명령 (기본: pick up the cup): ").strip()
        task = task if task else "pick up the cup"
        step4_inference(model_path=model, task=task)

    elif choice == "5":
        task = input("  작업 명령 (기본: pick up the cup): ").strip()
        task = task if task else "pick up the cup"
        steps = input("  스텝 수 (기본: 100): ").strip()
        steps = int(steps) if steps else 100
        quick_test(task=task, max_steps=steps)

    elif choice == "q":
        print("  종료")

    else:
        print("  잘못된 입력!")


if __name__ == "__main__":
    main()