def print_closest_cosine_words(vec, n=5):
    dists = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0), dim=1)
    lst = sorted(enumerate(dists.numpy()), key=lambda x: -x[1])
    for idx, similarity in lst[1:n+1]:
        print(glove.itos[idx], "\t%5.2f" % similarity)
