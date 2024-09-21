def train_sgns(textlist, window=5, embedding_size=8):
    # Create Training Data
    X, T, Y = tokenize_and_preprocess_text_with_frequency_cutoff(textlist, w2i, window)
    vocab_size = len(w2i)

    # Split the training data
    X_tensor = torch.tensor(X, dtype=torch.long)
    T_tensor = torch.tensor(T, dtype=torch.long)
    Y_tensor = torch.tensor(Y, dtype=torch.int)

    # Split data 80/20
    X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
        X_tensor, T_tensor, Y_tensor, test_size=0.2
    )

    # Instantiate the network & set up the optimizer
    network = SkipGramNegativeSampling(vocab_size, embedding_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
    
    def loss_function(predictions, labels):
        total_loss = 0
        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            if label == 1:  
                total_loss += -torch.log(torch.sigmoid(pred) + 1e-5)
            else:  
                total_loss += -torch.log(torch.sigmoid(-pred) + 1e-5)
        
        return total_loss.mean()

    # Training loop
    num_epochs = 30
    batch_size = 4

    # Lists to store loss for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        network.train()

        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            T_batch = T_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = network(X_batch, T_batch)

            # Compute the loss
            loss = loss_function(predictions, Y_batch)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss / len(X_train))

        # Validation
        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_val_batch = X_val[i:i + batch_size]
                T_val_batch = T_val[i:i + batch_size]
                Y_val_batch = Y_val[i:i + batch_size]

                predictions = network(X_val_batch, T_val_batch)
                loss = loss_function(predictions, Y_val_batch)
                val_loss += loss.item()

        val_losses.append(val_loss / len(X_val))

        # Print loss for every epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return network