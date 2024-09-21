def train_word2vec(textlist, window, embedding_size):
    # Set up a model with Skip-gram (predict context with word)
    # textlist: a list of the strings

    # Create the training data
    # this below only works if the prepare_texts is ran before hand and v2i is a global variable
    vocab_size = len(set(textlist))
    X, Y = tokenize_and_preprocess_text(textlist, v2i, window)

    # Split the training data
    # modified from the asnwer in this post here: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    X_tensor = torch.tensor(X, dtype=torch.long)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    X_train_tensor, X_val_tensor, Y_train_tensor, Y_val_tensor = train_test_split(
        X_tensor, Y_tensor, test_size=0.2, random_state=42
    )

    # instantiate the network & set up the optimizer

    network = Word2vecModel(vocab_size, embedding_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
    loss_function = torch.nn.CrossEntropyLoss()

    # training loop
    # training loop was modified from tutorial 5 in APS360 notes from 2022

    epochs = 50
    batch_size = 4
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        network.train()
        total_loss = 0

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_Y = Y_train_tensor[i:i + batch_size]

            optimizer.zero_grad()

            # Forward pass
            logits, _ = network(batch_X)
            loss = loss_function(logits, batch_Y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / (len(X_train_tensor) // batch_size)
        train_losses.append(avg_train_loss)

        # Val loss
        network.eval()
        with torch.no_grad():
            val_logits, _ = network(X_val_tensor)
            val_loss = loss_function(val_logits, Y_val_tensor)
            val_losses.append(val_loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")

    # Plotting the loss curves
    # Plot taken from tutorial 2 in APS360 from 2022
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return network