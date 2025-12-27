import pygame
import numpy as np

from src.reinforcement_learning.environments.grid_world import GridCellType, BasicActions

class AgentLearningVisualizer:
    def __init__(self, agent, step_interval = 10, cell_size_in_pixel=100, draw_policy_and_values=True, draw_visit_counts=False):
        self.agent = agent
        self.cell_size = cell_size_in_pixel
        self.border_color = (0, 0, 0) #Black
        self.player_color = (255, 255, 0) #Yellow
        self.color = {
            GridCellType.EMPTY: (192, 192, 192), #Light Grey
            GridCellType.FOOD: (0, 255, 0), #Green
            GridCellType.OBSTACLE: (45, 45, 45) #Dark Grey
        }
        self.step_interval = step_interval
        self.policy_symbols = {
            BasicActions.STAY: "-",
            BasicActions.UP: "^",
            BasicActions.DOWN: "v",
            BasicActions.LEFT: "<",
            BasicActions.RIGHT: ">"
        }
        self.draw_policy_and_values = draw_policy_and_values
        self.draw_visit_counts = draw_visit_counts


    def start(self):
        pygame.init()
        screen = pygame.display.set_mode((self.agent.environment.width * self.cell_size, self.agent.environment.height * self.cell_size))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 28)
        running = True
        last_step_time = pygame.time.get_ticks()
        cumulative_obj_reward = 0
        cumulative_subj_reward = 0

        while running:
            now = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if now - last_step_time >= self.step_interval:
                _, _, _, obj_reward, subj_reward = self.agent.act_one_step()
                cumulative_obj_reward += obj_reward
                cumulative_subj_reward += subj_reward
                last_step_time = now

            screen.fill("black")

            self.draw_grid(screen)
            self.draw_player(screen)
            if self.draw_policy_and_values:
                self._draw_policy(screen, font)

            if self.draw_visit_counts:
                self._draw_visit_counts(screen, font)

            steps_label = font.render(f"Steps: {self.agent.steps_completed}", True, "red")
            obj_reward_label = font.render(f"Sum Reward: {round(cumulative_obj_reward, 2)}", True, "red")
            subj_reward_label = font.render(f"Sum Happiness incl. Pain: {round(cumulative_subj_reward, 2)}", True, "red")
            current_pain_label = font.render(f"Current Pain: {round(self.agent.get_current_pain(), 4)}", True, "red")
            screen.blit(steps_label, (10, 0))
            screen.blit(obj_reward_label, (10, 20))
            screen.blit(subj_reward_label, (10, 40))
            screen.blit(current_pain_label, (10, 60))

            pygame.display.flip()
            clock.tick()

        pygame.quit()

    def draw_grid(self, screen):
        for pos, cell_type in np.ndenumerate(self.agent.environment.grid):
            self._draw_cell(screen, pos, cell_type)

    def draw_player(self, screen):
        center = pygame.Vector2(self.cell_size * (self.agent.get_agent_position()[0] + 0.5), self.cell_size * (self.agent.get_agent_position()[1] + 0.5))
        decrease_from_pain = (self.agent.get_current_pain() / self.agent.get_max_pain()) * 255
        color_including_pain = (max(0, self.player_color[0]), max(0, self.player_color[1] - decrease_from_pain), max(0, self.player_color[2] - decrease_from_pain))
        pygame.draw.circle(screen, color_including_pain, center, self.cell_size / 2 * 0.8)

    def _draw_cell(self, screen, pos, cell_type: GridCellType):
        color = self.color[cell_type]
        rect = pygame.Rect(pos[0] * self.cell_size, pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, self.border_color, rect, width=1)

    def _draw_policy(self, screen, font):
        for pos, _ in np.ndenumerate(self.agent.environment.grid):
            self._draw_policy_for_cell(screen, font, pos)

    def _draw_policy_for_cell(self, screen, font, pos):
        action, q_value = self.agent.get_max_q_action_value_pair(pos)
        symbol = font.render(self.policy_symbols[action], True, "black")
        value = font.render(f"{round(q_value, 2)}", True, (0, 128, 255))
        symbol_render_pos = ((pos[0] + 0.5) * self.cell_size, (pos[1] + 0.3) * self.cell_size)
        value_render_pos = ((pos[0] + 0.4) * self.cell_size, (pos[1] + 0.7) * self.cell_size)
        screen.blit(symbol, symbol_render_pos)
        screen.blit(value, value_render_pos)

    def _draw_visit_counts(self, screen, font):
        for pos, _ in np.ndenumerate(self.agent.environment.grid):
            self._draw_visit_count_for_cell(screen, font, pos)

    def _draw_visit_count_for_cell(self, screen, font, pos):
        value = font.render(f"{int(self.agent.get_cell_visit_count(pos))}", True, (0, 0, 0))
        value_render_pos = ((pos[0] + 0.4) * self.cell_size, (pos[1] + 0.3) * self.cell_size)
        screen.blit(value, value_render_pos)






