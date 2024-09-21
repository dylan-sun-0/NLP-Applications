def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []

    # Tokenize the input

    tokenized_corpus = [w2i[word] for word in textlist]

    # Loop through each token
    all_words = set(tokenized_corpus)

    for i, target_word in enumerate(tokenized_corpus):
        # find indexes 
        # method of indexing from chatGPT search
        start_index = max(0, i - window // 2)
        end_index = min(len(tokenized_corpus), i + window // 2 + 1)
        
        context_words = tokenized_corpus[start_index:i] + tokenized_corpus[i + 1:end_index]
        
        # Add positive sample
        for context_word in context_words:
            X.append(target_word)
            T.append(context_word)
            Y.append(1) 
            
        # Generate negative samples
        num_negative_samples = len(context_words)
        
        for _ in range(num_negative_samples):
            # assume random word can't be itself or a context word, as per instruction
            negative_sample = random.choice(list(all_words))
                
            X.append(target_word)
            T.append(negative_sample)
            Y.append(-1)
            
    return X, T, Y