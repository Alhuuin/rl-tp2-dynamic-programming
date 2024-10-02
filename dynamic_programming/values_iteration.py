import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION

    for _ in range (max_iter):
        new_values = np.copy(values)
        
        # Pour chaque état
        for state in range(mdp.observation_space.n):
            state_values = []  # Liste pour stocker les valeurs des actions possibles
            
            # Pour chaque action
            for action in range(mdp.action_space.n):
                # Obtention des données de la transition
                next_state, reward, _ = mdp.P[state][action]
                
                state_value = reward + gamma * values[next_state]
                state_values.append(state_value)
            
            # Mise à jour de la valeur de l'état avec la meilleure action (valeur maximale)
            new_values[state] = max(state_values)
        
        # Utilisation de np.allclose pour vérifier si les nouvelles valeurs sont proches des anciennes
        if np.allclose(values, new_values):
            break
        
        values = new_values  # Mettre à jour les valeurs pour la prochaine itération
    
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        delta = 0
        new_values = np.copy(values)

        # Pour chaque case de la grille:
        for row in range(env.height):
            for col in range(env.width):

                # On n'évalue pas les états terminaux ou les murs
                if env.grid[row, col] in {"P", "N", "W"}:
                    continue

                # Initialisation des variables pour la mise à jour de la valeur de l'état
                env.set_state(row, col)
                max_value = float("-inf") # - infini pour être sûr de ne pas être un faux maximum

                # Pour chaque action:
                for action in range(env.action_space.n):
                    next_state, reward, _ , _ = env.step(action, make_move=False)

                    next_row, next_col = next_state
                    value = reward + gamma * values[next_row, next_col]

                    max_value = max(max_value, value)

                # Mise à jour de la valeur de l'état (row, col)
                new_values[row, col] = max_value

                # Calcul du changement maximal (delta)
                delta = max(delta, np.abs(new_values[row, col] - values[row, col]))

        values = new_values

        # Vérification de la convergence
        if delta < theta:
            break

    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        delta = 0
        prev_values = values.copy()
        
        # Pour chaque position
        for row in range(env.height):
            for col in range(env.width):
                # On ignore les murs et les positions gagnantes ou perdantes (donc finales)
                if env.grid[row, col] in ['W', 'P', 'N']:
                    continue
                
                # Initialisation de la nouvelle position
                env.set_state(row, col)
                
                # Mise à jour des valeurs et de delta
                delta = value_iteration_per_state(
                    env, values, gamma, prev_values, delta
                )
        
        # On vérifie la convergence
        if delta < theta:
            break
    
    return values
    # END SOLUTION
