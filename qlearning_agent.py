from typing import Any, List
import numpy as np

from mlagents_envs.base_env import ActionTuple


class QLearning():

    def __init__(self, choose_action_method: str, initial_epsilon: float,
                 final_epsilon: float, gamma: float, learning_rate: float,
                 num_actions: int, num_episodes: int):
        self.choose_action_method: str = choose_action_method
        self.epsilon: float = initial_epsilon
        self.epsilon_decay: float = (initial_epsilon -
                                     final_epsilon) / num_episodes
        self.gamma: float = gamma
        self.learning_rate: float = learning_rate
        self.num_actions: int = num_actions
        self.num_episodes: int = num_episodes
        self.q_table: dict = {}
        self.cumulative_rewards: List[float] = []

    # Método de seleção de ação aleatória
    def choose_action_randomly(self, _: Any) -> int:
        return np.random.choice(self.num_actions)

    # Método de seleção de ação epsilon-greedy
    def choose_action_epsilon_greedy(self, state: int) -> int:
        ## Implementar código para realizar tal seleção
        return

    # Método que controla a política de seleção de ação do agente (aleatória ou epsilon-greedy)
    def choose_action(self, state: int) -> ActionTuple:
        action = self.choose_action_randomly(
            state) if self.choose_action_method.upper(
            ) == "RANDOM" else self.choose_action_epsilon_greedy(state)
        return ActionTuple(discrete=np.array([[action]]))

    # Método que realiza a atualização do par (estado, ação) na Tabela Q
    def update_q_table(self, current_state: int, action_tuple: ActionTuple,
                       reward: float, next_state: int) -> None:
        # Recupera a ação executada (0, 1 ou 2)
        action = action_tuple.discrete[0][0]

        # Inicializa o estado e ação na Tabela Q caso ainda não existam nela
        if current_state not in self.q_table:
            self.q_table[current_state] = {}
        if action not in self.q_table[current_state]:
            self.q_table[current_state][action] = 0

        # Cálculo de atualização do par (estado, ação)
        ## Implementar código para realizar tal atualização

    # Método que atualiza o valor do epsilon
    def update_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay

    # Método que atualiza a lista de recompensas cumulativas
    def add_cumulative_rewards(self, cum_rewards: float) -> None:
        self.cumulative_rewards.append(cum_rewards)

    # Método que calcula a média de recompensa cumulativa
    def get_mean_cumulative_rewards(self) -> float:
        return np.array(self.cumulative_rewards).mean()
