import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Router(nn.Module):
    def __init__(self, n_embed: int, n_experts: int, top_k: int = 2):
        super().__init__()

        self.top_k = top_k
        self.n_experts = n_experts
        self.noise_net = nn.Linear(input_dims=n_embed, output_dims=n_experts)
        self.route_network = nn.Linear(input_dims=n_embed, output_dims=n_experts)

    def __call__(self, x):
        B, T, _ = x.shape
        # router output
        logits = self.route_network(x)
        noise_logits = self.noise_net(x)
        # we must mask the the output for the other experts
        sparse = np.full_like(logits, float("-inf"))
        # gausssian noise for load balancing
        # gaussian noise * softplus(noise network)
        noise = mx.random.normal([B, T, self.n_experts]) * nn.Softplus()(noise_logits)
        # get indices of top k values
        top_k_indices = np.argsort(logits + noise, kind="quicksort", axis=-1)[
            :, :, -self.top_k :
        ]
        # get top k values
        topk = mx.topk(logits + noise, self.top_k)

        np.put_along_axis(sparse, top_k_indices, topk, axis=-1)
        sparse = mx.softmax(mx.array(sparse), axis=-1)

        return sparse, mx.array(top_k_indices)


# B, T, C = 1, 8, 32
# trans = Router(n_embed=C, n_experts=4)
# x = mx.random.normal((B, T, C))
# print(trans(x))
