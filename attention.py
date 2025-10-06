import torch
import torch.nn as nn

from data.test_inouts import TEST_INPUTS as inputs


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionMasked(nn.Module):
    """
    Use -inf to mask out the future tokens instead of zeroing them out. This is because softmax(-inf) = 0.
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
        context_vec = attn_weights @ values
        return attn_scores, attn_weights, context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  #1 Compared to the previous SelfAttention_v1 class, we added a dropout layer.
        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )  #2 The register_buffer call is also a new addition (more information is provided in the following text).

    def forward(self, x):
        b, num_tokens, d_in = x.shape  #3 We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   
        attn_scores.masked_fill_( #4 In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies.
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                 d_in, d_out, context_length, dropout, qkv_bias
             ) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads    # Reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)    # Uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)         # Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x)    # Tensor shape: (b, num_tokens, d_out)
        values = self.W_value(x)     # Tensor shape: (b, num_tokens, d_out)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       # We implicitly split the matrix by adding a num_heads dimension. Then we unroll the last dim: (b, num_tokens, d_out) -&gt; (b, num_tokens, num_heads, head_dim).
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        keys = keys.transpose(1, 2)          # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)    # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)      # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)

        attn_scores = queries @ keys.transpose(2, 3)   # Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    # Masks truncated to the number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf)     # Uses the mask to fill attention scores

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   # Tensor shape: (b, num_tokens, n_heads, head_dim)
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)    # Adds an optional linear projection
        return context_vec


if __name__ == "__main__":
    d_in = inputs.shape[1]      # The input embedding size, d=3
    d_out = 2 
    torch.manual_seed(123)
    # sa_v1 = SelfAttention_v1(d_in, d_out)
    # print(sa_v1(inputs))


    # torch.manual_seed(789)
    # sa_v2 = SelfAttention_v2(d_in, d_out)
    # print(sa_v2(inputs))

    # # Calculate attention weights
    # queries = sa_v2.W_query(inputs)
    # keys = sa_v2.W_key(inputs) 
    # attn_scores = queries @ keys.T
    # attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    # print(attn_weights)

    # """
    # Create a simple lower triangular mask. To be replace with a more complex mask later.
    # """
    # # Create a simple lower triangular mask
    # context_length = attn_scores.shape[0]
    # mask_simple = torch.tril(torch.ones(context_length, context_length))
    # print(mask_simple)

    # # Apply the mask to the attention weights
    # masked_simple = attn_weights * mask_simple
    # print(masked_simple)

    # # Re-normalize the masked attention weights
    # row_sums = masked_simple.sum(dim=-1, keepdim=True)
    # masked_simple_norm = masked_simple / row_sums
    # print(masked_simple_norm)

    # sa_masked = SelfAttentionMasked(d_in, d_out)
    # attn_scores, attn_weights, context_vecs = sa_masked(inputs)
    # print(attn_scores, '\n')
    # print(attn_weights, '\n')
    # print(context_vecs)

    # dropout = torch.nn.Dropout(0.5)
    # example = torch.ones(6, 6)
    # print(dropout(example))

    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    # print(batch.shape)  
    # context_length = batch.shape[1]
    # ca = CausalAttention(d_in, d_out, context_length, 0.0)
    # context_vecs = ca(batch)
    # print("context_vecs.shape:", context_vecs.shape)

    # Multi-head attention wrapper
    # context_length = batch.shape[1] # This is the number of tokens
    # d_in, d_out = 3, 2
    # mha = MultiHeadAttentionWrapper(
    #     d_in, d_out, context_length, 0.0, num_heads=2
    # )
    # context_vecs = mha(batch)

    # print(context_vecs)
    # print("context_vecs.shape:", context_vecs.shape)

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)