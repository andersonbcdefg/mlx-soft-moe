import mlx.core as mx
import mlx.nn as nn
import numpy as np


def relu_squared(x: mx.array):
    return mx.square(mx.maximum(x, 0))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def __call__(self, x):
        x = self.fc1(x)
        x = relu_squared(x)
        x = self.fc2(x)
        return x
    
class GroupedConv1D


class SoftMoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, hidden_dim=256):
        super(SoftMoE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.gate_fc = nn.Dense(in_dim, num_experts)
        self.expert_fc = nn.Dense(in_dim, hidden_dim)
        self.expert_out_fc = nn.Dense(hidden_dim, out_dim)

    def forward(self, x):
        # gate
        gate = self.gate_fc(x)
        gate = mx.softmax(gate, axis=1)

        # expert
        expert = self.expert_fc(x)
        expert = mx.relu(expert)
        expert = self.expert_out_fc(expert)

        # weighted sum
        y = mx.batch_dot(gate, expert)
        return y