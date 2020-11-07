import torch.optim as optim


def train(trainloader, val_loader, net, dev):
    #define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # train/val epoch_loss_history
    train_epoch_loss_history = []
    val_epoch_loss_history = []

    trainloader_iter = iter(trainloader)
    val_loader_iter = iter(val_loader)

    #training loop
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        # batches_per_epoch = len(dataset) // BATCH_SIZE # useless

        for phase in ['train', 'val']:
          start = time.time()
          running_loss = 0.0

          if phase == 'train':
            net.train(True) # Set model to training mode
          else:
            net.train(False) # Set model to evaluate mode

          print('Phase:', phase)

          loader = trainloader if phase == 'train' else val_loader
  
          for i, data in enumerate(loader, 0):
              # get the inputs; data is a list of [inputs, labels]
              # delete after
              if phase == 'train':
                inputs, labels = data #next(trainloader_iter)
                inputs, labels = inputs.to(dev), labels.to(dev)
              else:
                inputs, labels = data #next(val_loader_iter)
                inputs, labels = inputs.to(dev), labels.to(dev)

              #print('Input shape:', inputs.shape)
              #print('Epoch', epoch, ': [%d, %d]' % (i, batches_per_epoch))
              # zero the parameter gradients
              optimizer.zero_grad()

              # forward + backward + optimize
              outputs = net(inputs)
              loss = criterion(outputs, labels)
              if phase == 'train':
                loss.backward()
                optimizer.step()

              # print statistics
              running_loss += loss.item()

          mean_epoch_loss = running_loss / (i+1)
          if phase == 'train':
            train_epoch_loss_history.append(mean_epoch_loss)
          else:
            val_epoch_loss_history.append(mean_epoch_loss)
          print('Phase %s: [%d, %5d] loss: %.3f' %
              (phase, epoch + 1, i + 1, mean_epoch_loss), end="")
          end = time.time()
          print(" -- time elapsed (in sec):", end - start)

    print('Finished Training')
    return (train_epoch_loss_history, val_epoch_loss_history)


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 20

    train_transform = transforms.Compose([transforms.CenterCrop(CROP_SIZE),
                                            transforms.Resize((256, 256)),
                                            transforms.ToTensor()])

    full_dataset = SurgVisDataset(PATH_PORCINE_1, transform=train_transform, verbose=False)

    n = len(dataset)
    n_train = int(n * 0.90)
    n_val = int(n * 0.1)
    print("dataset len:", n)


    train_set, val_set, test_set = random_split(full_dataset, (n_train, n_val, 0))

    print("Train set:", len(train_set))
    print("Val set:", len(val_set))
    print("Test set", len(test_set))

    dataloader_train = DataLoader(train_set, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_set, batch_size=32, shuffle=False)

    model = Net()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('dev:', dev)
    model.to(dev)

    train_history, val_history = train(dataloader_train, dataloader_val, model, dev)