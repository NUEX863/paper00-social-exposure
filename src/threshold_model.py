"""Implement synchronous threshold-2 adoption and check small examples."""

import networkx as nx


def simulate_threshold2(graph, initial_adopters):
    """Return adoption counts from the initial state to the final growth round."""
    adopted = set(initial_adopters)
    if not adopted.issubset(graph.nodes):
        raise ValueError("Every initial adopter must exist in the graph.")

    history = [len(adopted)]

    while True:
        new_adopters = set()

        # Evaluate every candidate using the state at the start of the round.
        for node in graph.nodes:
            if node in adopted:
                continue

            adopted_neighbor_count = sum(
                neighbor in adopted for neighbor in graph.neighbors(node)
            )

            if adopted_neighbor_count >= 2:
                new_adopters.add(node)

        # Stop without appending a redundant final no-growth state.
        if not new_adopters:
            break

        # Apply all new adoptions together after evaluating every candidate.
        adopted.update(new_adopters)
        history.append(len(adopted))

    return history


if __name__ == "__main__":
    # Node 2 connects to both seeds; node 3 needs node 2 to adopt first.
    toy_graph = nx.Graph()
    toy_graph.add_edges_from([(0, 2), (1, 2), (0, 3), (2, 3)])

    one_seed = simulate_threshold2(toy_graph, {0})
    two_seeds = simulate_threshold2(toy_graph, {0, 1})

    print("Threshold-2 one-seed counts:", one_seed)
    print("Threshold-2 two-seed counts:", two_seeds)

    # Check that one neighbor is insufficient and updates are synchronous.
    assert one_seed == [1], "One adopted neighbor incorrectly triggered adoption."
    assert two_seeds == [2, 3, 4], "Threshold or synchronous update check failed."
    print("Threshold-2 small-network checks: PASS")
