from abc import ABC, abstractmethod

class NexodimRobot(ABC):
    """NxD 표준 규격 - 모든 로봇은 이걸 따라야 함"""

    @abstractmethod
    def connect(self):
        """로봇 연결"""
        pass

    @abstractmethod
    def get_observation(self):
        """현재 상태 읽기"""
        pass

    @abstractmethod
    def send_action(self, action):
        """로봇 움직이기"""
        pass

    @abstractmethod
    def disconnect(self):
        """연결 해제"""
        pass

    @abstractmethod
    def set_home_position(self):
        """현재 위치를 home position으로 저장"""
        pass

    @abstractmethod
    def go_home(self):
        """저장된 home position 위치로 부드럽게 이동"""
        pass

class NexodimPolicies(ABC):
    """NxD 표준 규격 - 모든 vla는 이걸 따라야 함"""

    @abstractmethod
    def load_policy(self):
        """load policy"""
        pass

    @abstractmethod
    def train_policy(self):
        """train policy"""
        pass

    @abstractmethod
    def validate_policy(self):
        """validate policy"""
        pass

    @abstractmethod
    def inference_policy(self, action):
        """run policy"""
        pass

    @abstractmethod
    def save_policy(self):
        """save policy"""
        pass