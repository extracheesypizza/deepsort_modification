from abc import ABC, abstractmethod
import numpy as np

class BaseReID(ABC):

    @abstractmethod
    def extract_features(self, crops):
        pass