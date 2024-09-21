def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    # edited function as per instructions in https://piazza.com/class/m09vx9jhhqx1sn/post/17
    # this method however does not work as expected and reproduces the identical words, so the original method was employed
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

def print_analgous_word(word1, word2, word3):
    print_closest_words(glove[word1] - glove[word2] + glove[word3], n=1)

# example from the table: plural verbs
print_analgous_word('work', 'works', 'speaks')

# 10 more examples using the same relationship
print('\n10 more words using the same relationship:')
print_analgous_word('work', 'works', 'runs')
print_analgous_word('work', 'works', 'walks')
print_analgous_word('work', 'works', 'rides')
print_analgous_word('work', 'works', 'leaves')
print_analgous_word('work', 'works', 'pushes')
print_analgous_word('work', 'works', 'starts')
print_analgous_word('work', 'works', 'eats')
print_analgous_word('work', 'works', 'reads')
print_analgous_word('work', 'works', 'probes')
print_analgous_word('work', 'works', 'taxes')