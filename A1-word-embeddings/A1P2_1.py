def method_a(category, test_word):
    sum = 0
    for word in category:
        sum += torch.cosine_similarity(glove[word].unsqueeze(0), glove[test_word].unsqueeze(0))
    return sum / len(category)

def method_b(category, test_word):
    sum = 0
    for word in category:
        sum += glove[word]
    avg = sum / len(category)
    return torch.cosine_similarity(avg.unsqueeze(0), glove[test_word].unsqueeze(0))

def compare_words_to_category(category, words):
    for word in test_words:
        # print the word then the results using method a and method b 
        print(word)
        print(method_a(color_words, word))
        print(method_b(color_words, word))
