from abc import abstractmethod, ABC
import numpy as np


class BasicAgent(ABC):
    @abstractmethod
    def sample_action(state: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def update() -> None:
        '''
        Updating net of agent
        '''
        pass