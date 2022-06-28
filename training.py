import os
from typing import Any, List, Tuple
import numpy as np

from mlagents_envs.registry import default_registry
from qlearning_agent import QLearning

ENV_NAME: str = "Basic"  # Ambiente Basic
NUM_ACTIONS: int = 3  # O ambiente Basic permite 3 ações
STATE_NORMALIZATION: int = 8  # Constante utilizada para tratar o estado no ambiente Basic

# Mapeamento das ações válidas do Basic
action_mapping: dict = {0: "Nothing", 1: "Left", 2: "Right"}


# Inicialização do ambiente no unity
def initialize_unity_env() -> Tuple[Any, Any]:
    env = default_registry["Basic"].make()
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    return env, behavior_name


# Inicialização do agente QLearning
def initialize_qlearning_agent() -> QLearning:
    choose_action_method = os.getenv("CHOOSE_ACTION_METHOD", "EPSILON_GREEDY")
    initial_epsilon = float(os.getenv("INITIAL_EPSILON", 1))
    final_epsilon = float(os.getenv("FINAL_EPSILON", 0.1))
    gamma = float(os.getenv("GAMMA", 0.5))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.5))
    num_episodes = int(os.getenv("NUM_EPISODES", 100))
    return QLearning(choose_action_method, initial_epsilon, final_epsilon,
                     gamma, learning_rate, NUM_ACTIONS, num_episodes)


# Método para realizar treinamento do agente QLearning
def training_qlearning_agent(env: Any, behavior_name: Any,
                             qlearning_agent: QLearning) -> None:
    for episode in range(qlearning_agent.num_episodes):
        # Inicialização do episódio
        env.reset()  # Reseta a posição do agente no ambiente
        decision_steps, terminal_steps = env.get_steps(
            behavior_name)  # Recupera as informações dos estados do ambiente
        tracked_agent = -1  # Indica que o agente ainda não foi identificado no ambiente
        done = False  # Indica que o episódio ainda não terminou
        episode_rewards = episode_num_actions = 0  # Inicializa a soma cumulativa de recompensas e o número de ações executadas no episódio

        # Enquanto o agente ainda não alcançou um estado terminal
        while not done:

            # Recupera a posição do agente no ambiente (para identificar em qual estado ele se encontra)
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Gera o número do estado correspondente à observação atual
            current_state = get_state(decision_steps[tracked_agent].obs)

            # O agente seleciona a ação a ser executada baseado no estado atual em que se encontra
            action = qlearning_agent.choose_action(current_state)

            # Execução da ação selecionada pelo agente no ambiente
            env.set_actions(behavior_name, action)
            env.step()
            episode_num_actions += 1

            # Atualiza as informações após a execução da ação do ambiente (novo estado, verifica se novo estado é terminal e recompensa imediatada)
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            next_state = get_state(
                decision_steps[tracked_agent].obs
            ) if tracked_agent in decision_steps else get_state(
                terminal_steps[tracked_agent].obs)
            done = True if tracked_agent in terminal_steps else False
            reward = decision_steps[
                tracked_agent].reward if tracked_agent in decision_steps else terminal_steps[
                    tracked_agent].reward
            episode_rewards += reward

            # Atualiação da Tabela Q
            qlearning_agent.update_q_table(current_state, action, reward,
                                           next_state)

        # Atualização do epsilon (considerado na estratégia de seleção de ação Epsilon-Greedy)
        qlearning_agent.update_epsilon()

        # Atualização da lista de recompensas cumulativas
        qlearning_agent.add_cumulative_rewards(episode_rewards)

        # Informações Relevantes do Episódio
        print(
            f"Total rewards for episode {episode} is {episode_rewards} - {episode_num_actions} executed actions - Epsilon: {qlearning_agent.epsilon} - Mean Cumulative Rewards: {qlearning_agent.get_mean_cumulative_rewards()}"
        )

    # Print da Tabela Q final
    print(f"Final Q Table: {qlearning_agent.q_table}")

    # Plot da média de recompensas cumulativas
    plot_mean_cumulative_rewards(qlearning_agent.cumulative_rewards)


# Transforma a observação em um número inteiro para facilitar trabalhar com a Tabela Q
def get_state(observation: Any) -> int:
    return np.where(observation[0] == 1)[0][0] - STATE_NORMALIZATION


# Plot do gráfico de evolução da média de recompensa cumulativa com relação ao tempo (número de episódios)
def plot_mean_cumulative_rewards(cumulative_rewards: List[float]) -> None:
    ## Implementar código para gerar gráfico de evolução da média de recompensa cumulativa
    return


if __name__ == "__main__":
    try:
        env, behavior_name = initialize_unity_env()
        qlearning = initialize_qlearning_agent()
        training_qlearning_agent(env, behavior_name, qlearning)
    finally:
        env.close()
        print("Closed environment")
