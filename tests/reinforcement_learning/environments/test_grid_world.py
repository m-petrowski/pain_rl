import pytest
import json

from src.reinforcement_learning.environments.grid_world import GridWorld, BasicActions

grid_size = 7
possible_starting_positions = [(0,6), (1, 6), (0, 5), (1, 5)]
obstacle_positions = [(3, 2), (3, 3), (3, 4), (2, 3), (4, 3), (3,0), (3, 6)]
possible_initial_food_positions = [(5, 0), (6, 0), (5, 1), (6, 1)]

@pytest.fixture
def basic_environment():
    """Creates a fresh instance of a Grid World before each test"""
    return GridWorld(grid_size, possible_starting_positions, possible_initial_food_positions, obstacle_positions, lifetime_learning=False, stationary=True)

def test_add_obstacles(basic_environment):
    for obstacle_pos in obstacle_positions:
        assert basic_environment.is_obstacle_position(obstacle_pos)

def test_initial_player_spawn(basic_environment):
    player_spawn = basic_environment.player.spawn_position
    assert player_spawn in possible_starting_positions

def test_initial_food_spawn(basic_environment):
    food_spawn = basic_environment.food_position
    assert food_spawn in possible_initial_food_positions

def test_overlapping_positions_throw_error():
    overlapping_starts = [(0,6), (1, 6)]
    overlapping_obstacles = [(0,5), (1, 6)]
    overlapping_food = [(0, 3), (1, 6)]
    with pytest.raises(ValueError, match="position cannot"):
        GridWorld(grid_size, overlapping_starts, possible_initial_food_positions, overlapping_obstacles)


    with pytest.raises(ValueError, match="overlap"):
        GridWorld(grid_size, overlapping_starts, overlapping_food, obstacle_positions)

    with pytest.raises(ValueError, match="position cannot"):
        GridWorld(grid_size, possible_starting_positions, overlapping_food, overlapping_obstacles)

def test_invalid_positions_throw_error():
    wrong_pos = [(-1, -1)]
    with pytest.raises(ValueError, match="grid"):
        GridWorld(grid_size, wrong_pos, possible_initial_food_positions, obstacle_positions)

    with pytest.raises(ValueError, match="grid"):
        GridWorld(grid_size, possible_starting_positions, wrong_pos, obstacle_positions)

    with pytest.raises(ValueError, match="grid"):
        GridWorld(grid_size, possible_starting_positions, possible_initial_food_positions, wrong_pos)

def test_from_param_map():
    json_param_map = """{
    "size": 7,
    "possible_starting_positions": [[0, 6], [1, 6], [0, 5], [1, 5]],
    "possible_initial_food_positions": [[5, 0], [6, 0], [5, 1], [6, 1]],
    "obstacle_positions": [[3, 2], [3, 3], [3, 4], [2, 3], [4, 3], [3,0], [3, 6]],
    "lifetime_learning" : false,
    "stationary": true,
    "food_location_change_steps": 1250
  }"""
    map = json.loads(json_param_map)
    GridWorld.from_param_map(map)


def test_step_method():
    env = GridWorld(grid_size, [(0, 6)], [(6, 0)], obstacle_positions, lifetime_learning=False, stationary=True)
    initial_position = env.get_player_position()
    assert initial_position == (0, 6)

    action = BasicActions.UP
    next_state, reward = env.step(action)
    assert next_state == (initial_position[0] + action.value[0], initial_position[1] + action.value[1])
    assert reward == 0

    action = BasicActions.DOWN
    next_state, reward = env.step(action)
    assert next_state == initial_position
    assert reward == 0

    action = BasicActions.LEFT
    next_state, reward = env.step(action)
    assert next_state == initial_position  # Player goes against border
    assert reward == 0

    action = BasicActions.RIGHT
    next_state, reward = env.step(action)
    assert next_state == (initial_position[0] + action.value[0], initial_position[1] + action.value[1])
    assert reward == 0

def test_step_method_with_reward_collection_lifetime():
    env = GridWorld(grid_size, [(5, 0)], [(6, 0)], obstacle_positions, lifetime_learning=True, stationary=True)
    action = BasicActions.RIGHT
    next_state, reward = env.step(action)
    assert next_state == (6, 0) #Teleported because of lifetime
    assert reward == 1

    action = BasicActions.STAY
    next_state, reward = env.step(action)
    assert next_state != (6, 0) #Teleported because of lifetime no matter the chosen action
    assert reward == 0


