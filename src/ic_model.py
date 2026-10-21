"""Implement synchronous independent cascade and check two boundary cases."""

import random

import networkx as nx


def simulate_ic(graph, initial_adopters, p, random_seed):
    """Return adoption counts from the initial state to the final growth round."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")

    adopted = set(initial_adopters)
    if not adopted.issubset(graph.nodes):
        raise ValueError("Every initial adopter must exist in the graph.")

    # Keep this simulation's random generator separate from other code.
    rng = random.Random(random_seed)

    # Only newly adopted nodes can attempt transmission in the next round.
    frontier = adopted.copy()
    history = [len(adopted)]

    while frontier:
        new_adopters = set()

        # Sort labels to keep the attempt order stable for our node types.
        for source in sorted(frontier):
            for target in sorted(graph.neighbors(source)):
                if target in adopted or target in new_adopters:
                    continue

                # Each eligible edge gets one independent transmission attempt.
                if rng.random() < p:
                    new_adopters.add(target)

        # Stop without appending a redundant final no-growth state.
        if not new_adopters:
            break

        # Apply all successful adoptions together at the end of the round.
        adopted.update(new_adopters)
        history.append(len(adopted))
        frontier = new_adopters

    return history


if __name__ == "__main__":
    # Use a three-node chain whose outcomes can be checked by hand.
    toy_graph = nx.path_graph(3)

    certain = simulate_ic(toy_graph, {0}, p=1.0, random_seed=42)
    impossible = simulate_ic(toy_graph, {0}, p=0.0, random_seed=42)

    print("IC p=1 adoption counts:", certain)
    print("IC p=0 adoption counts:", impossible)

    # Check propagation timing and the absence of spontaneous adoption.
    assert certain == [1, 2, 3], "Certain transmission check failed."
    assert impossible == [1], "Zero-probability check failed."
    print("IC boundary checks: PASS")

    from unittest.mock import Mock, patch

    # Provide a failing draw followed by a draw that must remain unused.
    controlled_rng = Mock()
    controlled_rng.random.side_effect = [0.8, 0.1]

    # Use one edge so there is exactly one eligible transmission attempt.
    one_edge = nx.path_graph(2)

    # Replace the random generator only during this controlled check.
    with patch.object(random, "Random", return_value=controlled_rng):
        failed_attempt = simulate_ic(
            one_edge, {0}, p=0.5, random_seed=42
        )

    print("IC failed-attempt adoption counts:", failed_attempt)
    print("IC transmission draw count:", controlled_rng.random.call_count)

    # A failed attempt must not be retried in a later round.
    assert failed_attempt == [1], "Failed transmission was incorrectly retried."
    assert controlled_rng.random.call_count == 1, "Unexpected extra attempt."
    print("IC one-attempt check: PASS")
