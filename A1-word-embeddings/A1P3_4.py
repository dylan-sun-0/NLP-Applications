class Word2vecModel(torch.nn.Module):
    # code partially taken from examples here: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers

        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)

        # prediction function takes embedding as input, and predicts which word in vocabulary as output

        self.output_layer = torch.nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        e = self.embeddings(x)
        logits = self.output_layer(e)
        
        return logits, e