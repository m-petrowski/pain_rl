import numpy as np

from enum import Enum
import random
from typing import List, Tuple


class GridCellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    FOOD = 2

class BasicActions(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

class Player:
    def __init__(self, spawn_position):
        self.spawn_position = spawn_position
        self.position = spawn_position
        self.is_finished = False

    def move(self, action: BasicActions):
        action_val = action.value
        self.position = (self.position[0] + action_val[0], self.position[1] + action_val[1])

class GridWorld:
    def __init__(self, size: int,
                 possible_starting_positions: List[Tuple[int, int]], # Should include the bottom left corner
                 possible_initial_food_positions: List[Tuple[int, int]], # Should include the top right corner
                 obstacle_positions: List[Tuple[int, int]], lifetime_learning=False, stationary=True, food_location_change_steps=1250):
        self.width = size
        self.height = size
        self.lifetime_learning = lifetime_learning
        self.stationary = stationary
        self.food_location_change_steps = food_location_change_steps
        self.reward = {
            GridCellType.EMPTY: 0,
            GridCellType.OBSTACLE: 0,
            GridCellType.FOOD: 1
        }

        if not self._are_valid_positions(obstacle_positions):
            raise ValueError("Not all obstacle positions are on the grid")
        if not self._are_valid_positions(possible_starting_positions):
            raise ValueError("Not all possible starting positions are on the grid")
        if not self._are_valid_positions(possible_initial_food_positions):
            raise ValueError("Not all possible initial food positions are on the grid")

        for pos in possible_starting_positions:
            if pos in obstacle_positions:
                raise ValueError("A possible starting position cannot be on an obstacle")

        for pos in possible_initial_food_positions:
            if pos in obstacle_positions:
                raise ValueError("A possible initial food position cannot be on an obstacle")

        if set(possible_starting_positions) & set(possible_initial_food_positions):
            raise ValueError("Possible starting positions and initial food positions overlap")

        self.possible_starting_positions = possible_starting_positions
        self.possible_initial_food_positions = possible_initial_food_positions
        self.obstacle_positions = obstacle_positions

        self.player = Player(spawn_position=random.choice(self.possible_starting_positions))
        self.grid = None
        self.visit_counts = None
        self.food_position = None
        self.initial_food_position = None
        self.steps_completed = 0
        self.initialize_grid()

    @classmethod
    def from_param_map(cls, param_map: dict):
        param_map["possible_starting_positions"] = [tuple(pos) for pos in param_map["possible_starting_positions"]]
        param_map["possible_initial_food_positions"] = [tuple(pos) for pos in param_map["possible_initial_food_positions"]]
        param_map["obstacle_positions"] = [tuple(pos) for pos in param_map["obstacle_positions"]]
        return cls(**param_map)


    def _are_valid_positions(self, positions: List[Tuple[int, int]]) -> bool:
        """
        Checks if all positions are on the grid
        :param positions: (list of tuples: [(int, int)...])
        :return: (bool)
        """
        for pos in positions:
            if not self._is_valid_position(pos):
                return False
        return True

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def _add_obstacle_positions_to_grid(self):
        """Adds initial obstacle positions to numpy grid"""
        for pos in self.obstacle_positions:
            self.grid[pos] = GridCellType.OBSTACLE

    def initialize_grid(self):
        """
        Initializes an empty numpy grid with obstacles and spawns initial food.
        """
        self.grid = np.full((self.width, self.height), GridCellType.EMPTY)
        self._add_obstacle_positions_to_grid()
        self._add_initial_food_to_grid()
        self.visit_counts = np.zeros((self.width, self.height))
        self._increase_visit_count(self.get_player_position())

    def _increase_visit_count(self, pos: Tuple[int, int]):
        """Increases the visit count of the visited grid position"""
        self.visit_counts[pos] = self.visit_counts[pos] + 1

    def _add_initial_food_to_grid(self):
        """Adds initial food position to numpy grid"""
        pos = random.choice(self.possible_initial_food_positions)
        self.grid[pos] = GridCellType.FOOD
        self.food_position = pos
        self.initial_food_position = pos

    def is_obstacle_position(self, pos: Tuple[int, int]) -> bool:
        return self.grid[pos] == GridCellType.OBSTACLE

    def is_food_position(self, pos: Tuple[int, int]) -> bool:
        return self.grid[pos] == GridCellType.FOOD

    def _spawn_new_food(self):
        """
        Removes the old food cell and spawn food on a new random corner which is not the current corner. The new food does not spawn in the same place as the old
        """
        old_food_pos = self.food_position
        self.food_position = self._get_new_food_position()
        self.grid[old_food_pos] = GridCellType.EMPTY
        self.grid[self.food_position] = GridCellType.FOOD

    def _get_new_food_position(self) -> Tuple[int, int]:
        """Helper method. Makes sure the new food position is not identical to the old"""
        pos = self._get_new_food_corner()
        while pos == self.food_position:
            pos = self._get_new_food_corner()

        return pos

    def _get_new_food_corner(self) -> Tuple[int, int]:
        """Helper method. Chooses a random food position from the corners"""
        possible_new_food_positions = [self.initial_food_position, (0, 0), (0, self.height - 1), (self.width - 1, self.height - 1)]
        return random.choice(possible_new_food_positions)


    def _get_random_empty_spawn_position(self) -> Tuple[int, int]:
        """Player could spawn in the same position as before"""
        pos = self._get_random_position_on_grid()
        while not self._valid_player_spawn(pos):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = (x, y)

        return pos

    def _get_random_position_on_grid(self) -> Tuple[int, int]:
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return x, y

    def _valid_player_spawn(self, pos: Tuple[int, int]) -> bool:
        """Helper method. Checks if the potential spawn position is empty"""
        return self.grid[pos] == GridCellType.EMPTY

    def step(self, action: BasicActions):
        if self.player.is_finished:
            self.player.is_finished = False

            if self.lifetime_learning:
                self.player.spawn_position = self._get_random_empty_spawn_position()
                self.player.position = self.player.spawn_position
                return self.player.position, self.reward[self.grid[self.player.position]]

        if not self.stationary:
            if self.steps_completed != 0 and self.steps_completed % self.food_location_change_steps == 0:
                self._spawn_new_food()


        old_pos = self.player.position
        action_tuple = action.value
        pos_after_action = (old_pos[0] + action_tuple[0], old_pos[1] + action_tuple[1])
        if self._is_valid_position(pos_after_action) and not self.is_obstacle_position(pos_after_action):
            self.player.position = pos_after_action
            if self.is_food_position(self.player.position):
                self.player.is_finished = True

        self.steps_completed += 1
        self._increase_visit_count(self.get_player_position())

        return self.player.position, self.reward[self.grid[self.player.position]]

    def get_player_position(self) -> Tuple[int, int]:
        return self.player.position

    def get_cell_visit_count(self,  pos: Tuple[int, int]) -> int:
        return self.visit_counts[pos]
