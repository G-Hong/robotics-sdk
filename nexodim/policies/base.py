from abc import ABC, abstractmethod

class NexodimModel(ABC):

    @abstractmethod
    def train(self, dataset_path):
        """학습"""
        pass

    @abstractmethod
    def validate(self, dataset_path):
        """검증"""
        pass

    @abstractmethod
    def predict(self, observation):
        """추론 - 관절값+이미지 받아서 행동 리턴"""
        pass