import random
import numpy as np
from tqdm import tqdm


def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None, neighbors_key=None, neighbors_time_key=None,
                            probabilities_key=None,
                            first_travel_key=None, quiet=False, use_linear=True, half_life=1):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]
            last_time = -np.inf

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:
                # For the first step
                if len(walk) == 1:
                    probabilities = d_graph[walk[-1]][first_travel_key]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                # probabilities = [1] * len(d_graph[walk[-1]].get(neighbors_key, []))

                walk_options = []
                for neighbor, p in zip(d_graph[walk[-1]].get(neighbors_key, []), probabilities):
                    times = d_graph[walk[-1]][neighbors_time_key][neighbor]
                    walk_options += [(neighbor, p, time) for time in times if time > last_time]

                # Skip dead end nodes
                if len(walk_options) == 0:
                    break

                if len(walk) == 1:
                    last_time = min(map(lambda x: x[2], walk_options))

                if use_linear:
                    time_probabilities = np.array(np.argsort(np.argsort(list(map(lambda x: x[2], walk_options)))[::-1])+1, dtype=np.float)
                    final_probabilities = time_probabilities*np.array(list(map(lambda x: x[1], walk_options)))
                    final_probabilities /= sum(final_probabilities)
                else:
                    final_probabilities = np.array(list(map(lambda x: np.exp(x[1]*(x[2]-last_time)/half_life), walk_options)))
                    final_probabilities /= sum(final_probabilities)

                walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=final_probabilities)[0]
                walk_to = walk_options[walk_to_idx]

                last_time = walk_to[2]
                walk.append(walk_to[0])

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
