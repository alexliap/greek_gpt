import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, n_embed: int, n_experts: int, top_k: int = 2):
        super().__init__()

        self.top_k = top_k
        self.n_experts = n_experts
        self.noise_net = nn.Linear(in_features=n_embed, out_features=n_experts)
        self.route_network = nn.Linear(in_features=n_embed, out_features=n_experts)

    def forward(self, x):
        # router output
        logits = self.route_network(x)
        noise_logits = self.noise_net(x)
        # gausssian noise for load balancing
        # gaussian noise * softplus(noise network)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


# B, T, C = 1, 8, 32
# trans = Router(n_embed=C, n_experts=4)
# x = torch.randn((B, T, C))
# print(trans(x))
