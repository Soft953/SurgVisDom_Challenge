from pathlib import Path
import time

import wandb

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from data_loader import SurgVisDataset
from model import Net


def train(trainloader, val_loader, model, dev, config):
    #define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    epoch_milestones = [int(config.epochs/2), int(config.epochs/2 + config.epochs/4)]
    scheduler = MultiStepLR(optimizer, milestones=epoch_milestones, gamma=0.1)

    train_loss_history = []
    val_loss_history = []

    # training/validation loop
    for epoch in range(config.epochs):
        start = time.time()
        training_loss = 0.0
        validation_loss = 0.0
        correct_training = 0
        correct_validation = 0

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
            correct_training += correct_per_batches

            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()

            # Log acc and loss per batches
            # wandb.log({"Train accuracy per batches": correct_per_batches / len(labels), "Train loss per batches": loss.item()})

        mean_training_loss = training_loss / (i+1)
        train_loss_history.append(mean_training_loss)
        end = time.time()
        print('Training step: [%d, %5d] loss: %.3f' % (epoch + 1, config.epochs, mean_training_loss), end="")
        print("| time elapsed (in sec):", end - start)
        # wandb.log({"Train accuracy per epoch": correct / len(trainloader.dataset), "Train loss per epoch": mean_training_loss})

        # Validation step
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
                correct_validation += correct_per_batches

                # print statistics
                validation_loss += loss.item()

                # Log acc and loss per batches
                # wandb.log({"Val accuracy per batches": correct_per_batches / len(labels), "Val loss per batches": loss.item()})

            mean_validation_loss = validation_loss / (i+1)
            if len(val_loss_history) and mean_validation_loss < np.min(val_loss_history):
                # Save model to wandb
                filename = "model_epoch_%d_loss_%.4f.h5" % (epoch + 1, mean_validation_loss)
                torch.save(model.state_dict(), filename)
                wandb.save(filename)
            val_loss_history.append(mean_validation_loss)
            end = time.time()
            print('Val step: [%d, %5d] loss: %.3f' % (epoch + 1, config.epochs, mean_validation_loss), end="")
            print("| time elapsed (in sec):", end - start)
            # Log everything once
            wandb.log({"Train accuracy per epoch": correct_training / len(trainloader.dataset),
                       "Train loss per epoch": mean_training_loss,
                       "Val accuracy per epoch": correct_validation / len(val_loader.dataset),
                       "Val loss per epoch": mean_validation_loss})

    print('Finished Training')
    return (train_loss_history, val_loss_history)



if __name__ == "__main__":
    wandb.init(project="surgvisdom")

    PATH = Path('train_1')
    PATH_PORCINE_1 = PATH.joinpath('Porcine')

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.batch_size = 128        # input batch size for training (default: 64)
    # config.test_batch_size = 10    # input batch size for testing (default: 1000)
    config.epochs = 50             # number of epochs to train (default: 10)
    config.lr = 0.1               # learning rate (default: 0.01)
    config.momentum = 0.1          # SGD momentum (default: 0.5) 
    config.no_cuda = False         # disables CUDA training
    config.seed = 42               # random seed (default: 42)
    # config.log_interval = 10     # how many batches to wait before logging training status
    config.crop_size = (420, 630)
    config.resize_shape = (256, 256)
    config.dataset_path = PATH_PORCINE_1

    train_transform = transforms.Compose([transforms.CenterCrop(config.crop_size),
                                            transforms.Resize(config.resize_shape),
                                            transforms.ToTensor()])

    dataset = SurgVisDataset(config.dataset_path, transform=train_transform, verbose=False)

    n = len(dataset)
    n_train = int(n * 0.80)
    n_val = int(n * 0.1)
    n_dev_test = int(n * 0.1)
    print("dataset len:", n)

    # Always split using the same seed
    torch.manual_seed(config.seed)
    train_set, val_set, dev_test_set = random_split(dataset, (n_train, n_val, n_dev_test))
    torch.manual_seed(torch.initial_seed())

    print("Train set:", len(train_set))
    print("Val set:", len(val_set))
    print("Test set", len(dev_test_set))

    model = Net()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('dev:', dev)
    model.to(dev)

    # Log metrics with wandb
    wandb.watch(model)

    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    train_history, val_history = train(dataloader_train, dataloader_val, model, dev, config)
