
"""
Module — 단일 실행 단위.
 
rule/VLA/RL 구분 없이 하나의 클래스로 통일한다.
실행 방식은 내부의 Executor가 결정하므로,
Module 자체는 "관측 → 추론 → context 반영 → 송출 → 완료 체크" 루프만 담당한다.
 
Module은 리프 노드이고, 여러 Module을 엮는 건 Wrapper의 책임이다.
Wrapper는 Module을 상속하므로 Module 자리에 들어갈 수 있다 (재귀 중첩).
"""


from __future__ import annotations

import time
from typing import Optional, Protocol, runtime_checkable, Any

from nexodim.utils.result import ModuleResult

"""Executor, Donchecker need modification!!!"""

# ══════════════════════════════════════════════════════
#  Executor 프로토콜 (느슨한 의존)
#
#  실제 Executor 구현은 executors/ 폴더에 있지만,
#  Module은 여기 정의된 인터페이스만 알면 된다.
#  runtime_checkable 덕분에 isinstance 체크도 가능.
# ══════════════════════════════════════════════════════
 
@runtime_checkable
class Executor(Protocol):
    """
    Module이 품는 '실행 방식' 부품.
    prepare → step(×N) → cleanup 생명주기를 가진다.
    """
 
    def prepare(self, context: Optional[ModuleResult]) -> None:
        """실행 시작 직전 호출. 모델 워밍업 등 일회성 준비 작업."""
        ...
 
    def step(
        self,
        obs: dict[str, Any],
        context: Optional[ModuleResult],
    ) -> tuple[dict[str, float], float]:
        """
        한 스텝의 action과 done_score를 생성.
        반환: (action dict, done_score 0~1)
        """
        ...
 
    def cleanup(self) -> None:
        """실행 종료 후 호출. 리소스 해제 등."""
        ...
 
 
# ══════════════════════════════════════════════════════
#  DoneChecker 프로토콜 (느슨한 의존)
#
#  실제 구현은 policies/vla/done_checker.py에 있지만,
#  Module은 update/is_stall/reset 세 메서드만 알면 된다.
# ══════════════════════════════════════════════════════
 
@runtime_checkable
class DoneChecker(Protocol):
    """done_score를 해석해 완료/stall 여부를 판정하는 부품."""
 
    def update(self, done_score: float) -> bool:
        """현재 스텝의 done_score로 완료 판정 (True면 종료 확정)."""
        ...
 
    def is_stall(self, trajectory: list[float]) -> bool:
        """지금까지의 궤적을 보고 stall 여부 판정."""
        ...
 
    def reset(self) -> None:
        """내부 상태 초기화 (consecutive 카운터 등)."""
        ...
 
 
class _NullDoneChecker:
    """
    done_checker를 주입받지 못했을 때 쓰는 기본값.
    update는 항상 False, is_stall도 항상 False.
    rule 기반 모듈처럼 done_score가 의미 없을 때 사용.
    """
 
    def update(self, done_score: float) -> bool:
        return False
 
    def is_stall(self, trajectory: list[float]) -> bool:
        return False
 
    def reset(self) -> None:
        pass

# ══════════════════════════════════════════════════════
#  Module 본체
# ══════════════════════════════════════════════════════
 
class Module:
    """
    단일 실행 단위.
 
    사용자는 보통 이 클래스를 직접 생성하지 않고,
    module.pick() 같은 팩토리나 Wrapper.pick() 같은 편의 메서드를 통해 사용한다.
    """
 
    def __init__(
        self,
        name: str,
        executor: Executor,
        done_checker: Optional[DoneChecker] = None,
        timeout: float = 15.0,
        fps: int = 30,
        lock_gripper_on_done: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            name: 모듈 식별자. AST 분석 매칭과 로그에 사용된다.
            executor: 실행 방식을 캡슐화한 Executor.
            done_checker: 완료 판정기. None이면 NullDoneChecker 사용.
            timeout: 최대 실행 시간(초). 넘기면 status=timeout.
            fps: 제어 루프 주기 (Hz).
            lock_gripper_on_done: 성공 완료 시 gripper를 잠글지.
            metadata: 결과의 metadata에 기본으로 병합될 값.
        """
        self.name = name
        self.executor = executor
        self.done_checker = done_checker or _NullDoneChecker()
        self.timeout = timeout
        self.fps = fps
        self.lock_gripper_on_done = lock_gripper_on_done
        self.base_metadata = metadata or {}
 
    # ══════════════════════════════════════════════
    #  run() — 모든 Module이 공유하는 공통 실행 루프
    # ══════════════════════════════════════════════
 
    def run(
        self,
        robot: Any,
        context: Optional[ModuleResult] = None,
    ) -> ModuleResult:
        """
        Module의 메인 진입점.
 
        Args:
            robot: NexodimRobot 인스턴스 (get_observation/send_action 제공).
            context: 직전 모듈의 결과. 첫 모듈이면 None.
 
        Returns:
            이 모듈의 실행 결과.
        """
        start_time = time.time()
        trajectory: list[float] = []
        last_action: dict[str, float] = {}
        status: str = "timeout"  # 기본값: 루프 탈출 못하면 timeout
 
        # Executor 준비
        self.executor.prepare(context)
        self.done_checker.reset()
 
        interval = 1.0 / self.fps
 
        try:
            while True:
                loop_start = time.time()
 
                # 1. 타임아웃 체크
                if loop_start - start_time >= self.timeout:
                    status = "timeout"
                    break
 
                # 2. 관측
                obs = robot.get_observation()
 
                # 3. Executor가 action과 done_score 생성
                action, done_score = self.executor.step(obs, context)
                trajectory.append(done_score)
 
                # 4. context의 gripper 잠금 반영
                #    (직전 모듈이 물건을 집고 넘어온 경우, 그리퍼 값을 덮어쓴다)
                if context is not None and context.has_gripper_lock():
                    action = dict(action)  # 원본 보호
                    action["gripper.pos"] = context.gripper_value
 
                # 5. 로봇 제어
                robot.send_action(action)
                last_action = action
 
                # 6. 완료 판정
                if self.done_checker.update(done_score):
                    status = "success"
                    break
                if self.done_checker.is_stall(trajectory):
                    status = "stall"
                    break
 
                # 7. 주기 맞추기
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
 
        except KeyboardInterrupt:
            status = "aborted"
        except Exception:
            # 예외는 cleanup 후 재발생시켜서 Wrapper가 처리하게 한다.
            status = "failed"
            self.executor.cleanup()
            raise
        finally:
            # KeyboardInterrupt나 정상 종료 시 cleanup 보장.
            if status != "failed":
                self.executor.cleanup()
 
        # ── 결과 조립 ──
        return self._build_result(
            robot=robot,
            status=status,
            trajectory=trajectory,
            last_action=last_action,
            duration=time.time() - start_time,
        )
 
    # ══════════════════════════════════════════════
    #  analyze() — AST 분석에서 호출되는 인터페이스
    # ══════════════════════════════════════════════
 
    def analyze(self) -> set[str]:
        """
        이 모듈이 사용할 하위 모듈 이름 집합.
        리프 노드인 Module은 자기 이름만 반환.
        Wrapper는 이 메서드를 오버라이드해서 재귀 수집한다.
        """
        return {self.name}
 
    # ══════════════════════════════════════════════
    #  내부 헬퍼
    # ══════════════════════════════════════════════
 
    def _build_result(
        self,
        robot: Any,
        status: str,
        trajectory: list[float],
        last_action: dict[str, float],
        duration: float,
    ) -> ModuleResult:
        """실행 종료 후 ModuleResult 조립."""
 
        # 종료 시점 관측 — 다음 Transition이 쓸 수 있게 robot_state로 남긴다.
        try:
            final_obs = robot.get_observation()
            robot_state = {
                k: float(v) for k, v in final_obs.items()
                if isinstance(k, str) and k.endswith(".pos")
            }
        except Exception:
            robot_state = {}
 
        # metadata 기본값 + 실행 정보 병합
        metadata = dict(self.base_metadata)
        metadata["module"] = self.name
 
        # gripper 잠금 여부는 '성공 + lock_gripper_on_done' 둘 다 True여야 함
        locked = bool(self.lock_gripper_on_done and status == "success")
        gripper_value = float(last_action.get("gripper.pos", 0.0)) if locked else 0.0
 
        return ModuleResult(
            status=status,  # type: ignore[arg-type]
            robot_state=robot_state,
            final_done_score=trajectory[-1] if trajectory else 0.0,
            done_score_trajectory=trajectory,
            gripper_locked=locked,
            gripper_value=gripper_value,
            metadata=metadata,
            duration=duration,
        )

