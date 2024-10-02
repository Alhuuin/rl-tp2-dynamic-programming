# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION

    # Initialisation des valeurs de base (nous utilisons n-4, donc nous avons besoin d'initialiser au moins jusqu'à 3)
    if n == 0:
        return 1
    elif n == 1 or n == 3:
        return 0
    elif n == 2:
        return 3

    # Formule de récurrence
    return 4 * domino_paving(n - 2) - domino_paving(n - 4)
    # END SOLUTION
