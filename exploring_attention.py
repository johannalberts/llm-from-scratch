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

#=========================
# SELF-ATTENTION
#=========================
x_2 = inputs[1]     # The second input element
d_in = inputs.shape[1]      # The input embedding size, d=3
d_out = 2         # The output embedding size, d_out=2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print("\nW_query:\n", W_query)

query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value
print("\nquery_2:\n", query_2)

# Get key and value vectors for all inputs
keys = inputs @ W_key 
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Compute attention score between query_2 and all keys_2
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# Vectorized computation of attention scores between query_2 and all keys
attn_scores_2 = query_2 @ keys.T       #1
print(attn_scores_2)

# Scaled dot-product attention weights
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# Compute context vector for input 2
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)