import torch

from data_loader import create_dataloader_v1


def main():
    with open("data/verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    torch.manual_seed(123) # set seed for reproducibility
    vocab_size = 50257 # vocab size for GPT-2
    output_dim = 256 # embedding dimension for testing
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # token embedding layer converting tokens ids to vectors of the output dimension

    max_length = 4 # sequence length for testing
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False # stride and max_length are equal to avoid overlap
    )
    data_iter = iter(dataloader) # create an iterator from the dataloader
    inputs, targets = next(data_iter) # get the first batch
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs) # get the token embeddings for the input token ids (a lookup table in essence)
    print("Token Embeddings:\n", token_embeddings[0])
    print("\nToken Embeddings shape:\n", token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) # absolute positional embedding layer
    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # get the positional embeddings for positions 0 to context_length-1
    print("\nPositional Embeddings shape:\n", pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings # add token and positional embeddings to create input embeddings
    print("\nInput Embeddings shape:\n", input_embeddings.shape)


if __name__ == "__main__":
    main()
