def train_2(trainloader, val_loader, model, dev, config):
    #define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    train_loss_history = []
    val_loss_history = []

    # training/validation loop
    for epoch in range(config.epochs):
        start = time.time()
        training_loss = 0.0
        validation_loss = 0.0
        correct = 0

        # Set the model to training mode
        model.train()
        
        #Training step loop
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Tensor to cpu or gpu
            inputs, labels = inputs.to(dev), labels.to(dev)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Compute accuracy
            pred = outputs.max(1, keepdim=True)[1]
            correct_per_batches = pred.eq(labels.view_as(pred)).sum().item()
            correct += correct_per_batches

            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()

            # Log acc and loss per batches
            wandb.log({"Train accuracy per batches": correct_per_batches / len(labels), "Train loss per batches" % (phase): loss.item()})

        mean_training_loss = training_loss / (i+1)
        train_loss_history.append(mean_training_loss)
        end = time.time()
        print('Training step: [%d, %5d] loss: %.3f' % (epoch + 1, config.epochs, mean_training_loss), end="")
        print("| time elapsed (in sec):", end - start)
        wandb.log({"Train accuracy per epoch" : correct / len(trainloader.dataset), "Train loss per epoch": mean_training_loss})

        # Validation step
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # Tensor to cpu or gpu
                inputs, labels = inputs.to(dev), labels.to(dev)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Compute accuracy
                pred = outputs.max(1, keepdim=True)[1]
                correct_per_batches = pred.eq(labels.view_as(pred)).sum().item()
                correct += correct_per_batches

                # print statistics
                validation_loss += loss.item()

                # Log acc and loss per batches
                wandb.log({"Val accuracy per batches": correct_per_batches / len(labels), "Val loss per batches" % (phase): loss.item()})

            mean_validation_loss = validation_loss / (i+1)
            if len(val_loss_history) and mean_validation_loss < np.min(val_loss_history):
                # Save model to wandb
                filename = "model_epoch_%d_loss_%.4f.h5" % (epoch + 1, mean)
                torch.save(model.state_dict(), filename)
                wandb.save(filename)
            val_loss_history.append(mean_validation_loss)
            end = time.time()
            print('Val step: [%d, %5d] loss: %.3f' % (epoch + 1, config.epochs, mean_validation_loss), end="")
            print("| time elapsed (in sec):", end - start)
            wandb.log({"Val accuracy per epoch" : correct / len(val_loader.dataset), "Val loss per epoch": mean_validation_loss})

    print('Finished Training')
    return (train_loss_history, val_loss_history)
