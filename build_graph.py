import torch
import collections


def construct_global_graph(seqs, window_size, K, padding_idx=1):
    # the dict to store the neighbors for each word
    neighbors = collections.defaultdict(list)

    # number of words in the sentence, including the pading word identified as 0
    num_of_words = -1

    # iterate over each source or target sentence
    for seq in seqs:
        # filter the padding words
        seq = list(filter(lambda w: w != padding_idx, seq))
        # finding the maximum word id
        num_of_words = max(num_of_words, *seq)
        # filtering the  padding word marked as zero
        for i, w in enumerate(seq[:-1]):
            # left neighbors
            # neighbors[w].extend(seq[max(i-window_size,0):i])

            # only considering outgoing neighbors
            neighbors[w].extend(seq[i + 1:i + window_size + 1])

    # construct the adjacent matrix
    adj = torch.zeros(num_of_words + 1, num_of_words + 1)

    # construct the neighborhood index
    neighbor_indices = torch.zeros(num_of_words + 1, K, dtype=torch.int64)

    # process the neighbors for each word by only keeping the K neighbors with the largest weight
    for k, nns in neighbors.items():
        cols, weights = list(zip(*collections.Counter(nns).most_common(K)))

        # the adjacent matrix
        adj[k, torch.LongTensor(cols)] = torch.FloatTensor(weights)

        # neighborhood index
        neighbor_indices[k] = torch.LongTensor(list(cols) + [padding_idx] * (K - len(cols)))

    # normalize the edge weights along the row direction
    return adj / (torch.sum(adj, dim=-1, keepdims=True) + 1e-05), neighbor_indices