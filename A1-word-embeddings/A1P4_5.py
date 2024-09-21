# Subsampling/Frequency cutoff taken from below locations:
# https://huggingface.co/docs/transformers/preprocessing
# https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/

def tokenize_and_preprocess_text_with_frequency_cutoff(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []

    # Tokenize the input

    tokenized_corpus = [w2i[word] for word in textlist]

    word_counts = Counter(tokenized_corpus)

    # Loop through each token
    all_words = set(tokenized_corpus)

    for i, target_word in enumerate(tokenized_corpus):
        # Define the context window
        start_index = max(0, i - window // 2)
        end_index = min(len(tokenized_corpus), i + window // 2 + 1)
        
        # Get context words within the window
        context_words = tokenized_corpus[start_index:i] + tokenized_corpus[i + 1:end_index]

        # Reduce the number of samples selected from window depending on word frequency
        if word_counts[target_word] > 100:
            num_samples = 1
        elif word_counts[target_word] > 50:
            num_samples = 3
        elif word_counts[target_word] > 10:
            num_samples = 5
        else:
            num_samples = len(context_words) 

        # Ensure num_samples does not exceed the available context words
        num_samples = min(num_samples, len(context_words))

        # Add positive samples
        for context_word in random.sample(context_words, num_samples):
            X.append(target_word)
            T.append(context_word)
            Y.append(1)
    
        # Generate negative samples
        num_negative_samples = len(context_words)
        
        for _ in range(num_negative_samples):
            negative_sample = random.choice(list(all_words))
            while negative_sample == target_word or negative_sample in context_words:
                negative_sample = random.choice(list(all_words))
                
            X.append(target_word)
            T.append(negative_sample)
            Y.append(-1)

    return X, T, Y