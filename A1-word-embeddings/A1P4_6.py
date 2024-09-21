class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)

    def forward(self, x, t):

        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.

        x_embeddings = self.embeddings(x)
        t_embeddings = self.embeddings(t)

        prediction = (x_embeddings * t_embeddings).sum(dim=1)

        return prediction