import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)

# Compute attention scores using dot product
query = inputs[1]
print("Query vector:\n", query)
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # Compute attention score between x_i and the query
print("Attention scores torch.dot:\n", attn_scores_2)

attn_scores_2m = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2m[i] = x_i @ query # Compute attention score between x_i and the query
print("Attention scores matmul:\n", attn_scores_2m)

# Alternative to compute attention scores using matrix multiplication
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query))

# Normalise attention scores to get weights
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("\nAttention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Naive softmax normalization as an alternative (preferred)
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("\nAttention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
