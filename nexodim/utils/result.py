from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ModuleStatus = Literal[
    "success",   # 정상 완료 — 일반적으로 다음 모듈로 진행
    "failed",    # 실행 실패 — Wrapper의 on_failure 정책에 따라 처리
    "timeout",   # 제한 시간 초과 — 기본적으로 실패로 취급
    "aborted",   # 외부 요인으로 중단 (KeyboardInterrupt 등)
    "stall",     # done_score 진전 없음 — 실패는 아니나 더 못 나아감
    "skipped",   # 조건에 의해 스킵됨 (예: probe 모드)
]

@dataclass
class ModuleResult:
    """
    모듈 실행의 결과 묶음.
 
    이 클래스는 순수 데이터 컨테이너이며, 로직은 편의 메서드 몇 개만 갖는다.
    각 필드의 책임은 아래와 같다.

    """

    # ── 실행 상태 ──
    # 모듈 자신이 결정. Wrapper가 분기 판단에 사용.
    status: ModuleStatus = "success"
 
    # ── 로봇 상태 ──
    # 모듈 종료 시점의 관절 상태. SO101.get_observation()의 형식과 동일.
    # 예: {"shoulder_pan.pos": -9.2, "gripper.pos": -30.0, ...}
    # Transition이 이 값을 그대로 send_action에 넘길 수 있도록 키 포맷을 유지한다.
    robot_state: dict[str, float] = field(default_factory=dict)
 
    # ── VLA 완료 신호 ──
    # 모듈 종료 시점의 done_score (0.0 ~ 1.0).
    # VLA 모듈만 의미 있는 값을 넣고, rule/RL 모듈은 0.0 그대로 둔다.
    final_done_score: float = 0.0
 
    # 실행 중 매 스텝 기록된 done_score 전체 궤적.
    # 디버깅, 재학습 데이터, handoff_threshold 판단에 쓰인다.
    done_score_trajectory: list[float] = field(default_factory=list)
 
    # ── 관절 합성용 ──
    # True면 다음 모듈의 Transition이 gripper를 이 값으로 잠가서 전달한다.
    # pick 모듈처럼 "잡은 상태를 유지해야 하는" 모듈이 True로 설정.
    gripper_locked: bool = False
    gripper_value: float = 0.0
 
    # ── 자유 메타데이터 ──
    # 모듈이 남기고 싶은 구조화되지 않은 정보.
    # 예: detect 모듈 → {"found": True, "object_pos": [x, y, z]}
    #     pick 모듈   → {"task": "pick", "object": "cup"}
    metadata: dict[str, Any] = field(default_factory=dict)
 
    # ── 프로파일링 ──
    # 실행에 소요된 시간 (초). 리소스 매니저와 성능 로깅에 쓰인다.
    duration: float = 0.0


    # ══════════════════════════════════════════════
    #  편의 메서드 — Wrapper의 body() 안에서 쓰기 쉽게
    # ══════════════════════════════════════════════
 
    def is_success(self) -> bool:
        """성공 상태 여부. Wrapper의 if문에서 사용."""
        return self.status == "success"
 
    def is_failure(self) -> bool:
        """실패/타임아웃/중단 등 부정적 종료인지."""
        return self.status in ("failed", "timeout", "aborted")
 
    def should_continue(self) -> bool:
        """
        이 결과를 보고 다음 모듈로 진행해도 되는지 판단.
        success와 stall은 진행 (stall은 실패는 아니고 단지 VLA 신호가 멎은 것).
        나머지는 중단.
        """
        return self.status in ("success", "stall", "skipped")
 
    def has_gripper_lock(self) -> bool:
        """다음 모듈에서 그리퍼를 강제로 유지해야 하는지."""
        return self.gripper_locked

    # ══════════════════════════════════════════════
    #  팩토리 — 자주 쓰는 케이스를 간결하게
    # ══════════════════════════════════════════════
 
    @classmethod
    def empty(cls) -> "ModuleResult":
        """첫 모듈 실행 전, context로 넘길 초기 결과."""
        return cls(status="success")
 
    @classmethod
    def skipped(cls, reason: str = "") -> "ModuleResult":
        """probe 모드 등에서 실제 실행을 건너뛸 때."""
        return cls(status="skipped", metadata={"skip_reason": reason})
 
    @classmethod
    def failed(cls, reason: str = "", **kwargs) -> "ModuleResult":
        """실패 결과를 간결하게 만드는 팩토리."""
        metadata = kwargs.pop("metadata", {})
        metadata["fail_reason"] = reason
        return cls(status="failed", metadata=metadata, **kwargs)
 
