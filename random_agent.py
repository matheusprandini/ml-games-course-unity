from mlagents_envs.registry import default_registry

ENV_NAME = "Basic"
NUM_EPISODES = 10

action_mapping = {
    0: "Nothing",
    1: "Left",
    2: "Right"
}

env = default_registry["Basic"].make()
env.reset()
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

def execute_random_agent():
    for episode in range(NUM_EPISODES):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1
        done = False
        episode_rewards = num_actions = 0
        while not done:

            # Recupera a posição do agente no ambiente (para identificar em qual estado ele se encontra)
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            # Escolhe ação de maneira aleatória
            action = spec.action_spec.random_action(len(decision_steps))

            # Execução da ação selecionada no ambiente
            env.set_actions(behavior_name, action)
            env.step()
            num_actions += 1

            # Atualiza as informações após a execução da ação do ambiente (novo estado, verifica se novo estado é terminal e recompensa imediatada)
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:
                episode_rewards += decision_steps[tracked_agent].reward
                print(f"Decision Action {num_actions}: {action_mapping[action.discrete[0][0]]} - Observation: {decision_steps[tracked_agent].obs} - Reward: {decision_steps[tracked_agent].reward}")
            if tracked_agent in terminal_steps: # O agente chegou a um estado terminal
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True
                print(f"Terminal Action {num_actions}: {action_mapping[action.discrete[0][0]]} - Observation: {terminal_steps[tracked_agent].obs} - Reward: {terminal_steps[tracked_agent].reward}")
        print(f"Total rewards for episode {episode} is {episode_rewards} - {num_actions} executed actions")

if __name__ == "__main__":
    try:
        execute_random_agent()
    finally:
        env.close()
        print("Closed environment")
