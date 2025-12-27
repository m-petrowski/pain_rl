from abc import ABC, abstractmethod
from typing import Tuple, Protocol


class VisualizableAgent(ABC):
    @abstractmethod
    def get_agent_position(self) -> Tuple[int, int]:
        """Returns position of the agent in the grid"""
        pass

    @abstractmethod
    def get_value_at_position(self, state: Tuple[int, int]) -> float:
        """Returns the state value at"""
        pass

    @abstractmethod
    def get_cell_visit_count(self, pos: Tuple[int, int]) -> int:
        """Returns the visit cound of the cell"""
        pass

class PainVisualizationAgent(ABC):
    @abstractmethod
    def get_max_pain(self):
        """Returns the maximum pain the agent can feel"""
        pass

    @abstractmethod
    def get_subjective_pain(self):
        """Returns the pain the agent feels multiplied by its pain weight (w4)"""
        pass

    @abstractmethod
    def get_current_pain(self):
        """Returns the pain the agent currently feels"""
        pass