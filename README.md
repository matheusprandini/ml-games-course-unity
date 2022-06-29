# ML-Games-Course-Unity

Código Base para o trabalho de Aprendizado por Reforço no Unity da disciplina Aprendizagem de Máquina aplicada a Jogos.

## Objetivo

Implementação de um agente treinado pelo método QLearning para o ambiente [Basic](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#basic).

## Repositório Base - ml-agents

https://github.com/Unity-Technologies/ml-agents

## Dependências

A instalação das dependências pode ser encontrada no repositório base, conforme orientado abaixo:

- https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md
- https://github.com/miyamotok0105/unity-ml-agents/blob/master/docs/Installation-Windows.md

## Configuração (Hiperparâmetros)

É necessário criar um arquivo ```.env``` no diretório raiz do projeto (ele contém as variáveis de ambiente necessárias para executar com sucesso). Ele é formado pelas variáveis abaixo (exemplo no arquivo ```.env.example```):

|   **Nome**   |  **Valor _Default_**  |    **Descrição**    |
| :---:        |     :---:      |          :---: |
|CHOOSE_ACTION_METHOD| RANDOM | Representa a política de seleção de ação do agente (pode assumir os valores RANDOM e EPSILON_GREEDY)    |
|INITIAL_EPSILON| 1 | Indica o valor inicial do hiperparâmetro epsilon (utilizado na política epsilon-greedy)      |
|FINAL_EPSILON| 0.1 | Indica o valor final do hiperparâmetro epsilon (utilizado na política epsilon-greedy) |
|GAMMA| 0.5 | Indica o valor do hiperparâmetro gamma (utilizado na atualização da Tabela Q) |
|LEARNING_RATE| 0.5 | Indica o valor do hiperparâmetro learning rate (utilizado na atualização da Tabela Q) |
|NUM_EPISODES| 100 | Indica o número de episódios utilizados no treinamento do agente |

## Execução do Projeto

### Agente Aleatório

Para validar o funcionamento do código e a instação das dependências, execute o script com um agente que executa apenas ações aleatórias no ambiente Basic a partir do diretório raiz do projeto: 

```python3 random_agent.py```

### Agente QLearning

Após completar o código no arquivo _qlearning.py_, basta executar o script de treinamento do agente baseado no método QLearning a partir do diretório raiz do projeto:

```python3 training.py```

Obs: exportar as envs antes de executar o comando acima.

## Material de Apoio 

Alguns materiais produzidos para estudo sobre o método QLearning podem ser encontrados em [Docs](docs/)
