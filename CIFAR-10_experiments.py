# -*- coding: utf-8 -*-
"""
 This code tries to implement the MixMatch technique from the [paper](https://arxiv.org/pdf/1905.02249.pdf) MixMatch: A Holistic Approach to Semi-Supervised Learning and recreate their results on CIFAR10 with WideResnet28.

 It depends on Pytorch, Numpy and imgaug. The WideResnet28 model code is taken from [meliketoy](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py)'s github repository. Hopefully I can train this on Colab with a Tesla T4. :)
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary
import sys

from mixmatch_utils import get_augmenter, mixmatch
from pytorchtools import EarlyStopping, ModelCheckpoint


# training_amount + training_u_amount + validation_amount <= 50 000


def basic_generator(x, y=None, batch_size=32, shuffle=True):
    i = 0
    all_indices = np.arange(len(x))
    if shuffle:
        np.random.shuffle(all_indices)
    while(True):
        indices = all_indices[i:i+batch_size]
        if y is not None:
            x_norm = x[indices] / 255
            x_norm = x_norm - np.mean(x[indices])
            x_norm = x_norm/(np.std(x_norm) + 1e-10)
            yield x_norm, y[indices]
        else:
            x_norm = x[indices] / 255
            x_norm = x_norm - np.mean(x[indices])
            x_norm = x_norm / (np.std(x_norm) + 1e-10)
            yield x_norm
        i = (i + batch_size) % len(x)

def mixmatch_wrapper(x, y, u, model, batch_size=32):
    augment_fn = get_augmenter()
    train_generator = basic_generator(x, y, batch_size)
    unlabeled_generator = basic_generator(u, batch_size=batch_size)
    while(True):
        xi, yi = next(train_generator)
        ui = next(unlabeled_generator)
        yield mixmatch(xi, yi, ui, model, augment_fn)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg=img
    #npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def L_u(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert input_logits.size() == target_logits.size()
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_logits, input_logits, reduction='mean')

def train_dl():

    # Data preprocessing and split
    training_amount = 300

    training_u_amount = 30000

    validation_amount = 10000

    transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    X_train = np.array(trainset.data)
    y_train = np.array(trainset.targets)

    X_test = np.array(testset.data)
    y_test = np.array(testset.targets)

    # Train set / Validation set split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)

    # Train unsupervised / Train supervised split
    # Train set / Validation set split
    X_train, X_u_train, y_train, y_u_train = train_test_split(X_train, y_train, test_size=training_u_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)

    X_remain, X_train, y_remain, y_train = train_test_split(X_train, y_train, test_size=training_amount, random_state=1,
                                                              shuffle=True, stratify=y_train)


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # DL related init variables

    # Hyper parameters
    epochs = int(1e3)
    num_classes = 10
    batch_size = 128
    # learning_rate = 1e-4
    learning_rate = 0.002
    min_lr = 1e-12



    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model definition
    model_filepath = 'weights.best.pt'
    # model = WideResNet(num_classes=num_classes).to(device)

    model = (models.SqueezeNet(num_classes=num_classes)).to(device)  # TODO: Define and use "Wide ResNet-28"


    # Training data generators
    # Data
    train_data_gen = mixmatch_wrapper(X_train, y_train, X_u_train, model, batch_size=batch_size)

    # Validation data generators
    val_data_gen = basic_generator(X_val, y_val, batch_size=batch_size, shuffle=True)

    # Model summary Keras style
    summary(model, (3, 150, 150))

    # Optimization parameters
    # Used criterions
    supervised_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    consistency_criterion = L_u
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.5,
                                                           patience=10,
                                                           verbose=True,
                                                           threshold=0.001,
                                                           threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=min_lr,
                                                           eps=1e-08)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=30, mode='max', verbose=True)
    checkpoint = ModelCheckpoint(checkpoint_fn=model_filepath, mode='max', verbose=True)

    # Training level variables

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_val_losses = []

    # to track the average training acc per epoch as the model trains
    avg_train_acc_s = []
    # to track the average validation acc per epoch as the model trains
    avg_val_acc_s = []

    # Prepare the model for train
    model.train()

    print('\n===== TRAINING =====\n')
    for epoch in range(epochs):

        # Epoch training

        # Epoch level variables
        # to track the training loss as the model trains
        train_losses_item = []
        # to track the validation loss as the model trains
        val_losses_item = []

        # to track the training acc as the model trains
        train_acc_s_item = []
        # to track the validation acc as the model trains
        val_acc_s_item = []

        avg_train_loss = 0
        avg_train_acc = 0
        total_train = 0
        correct_train = 0

        avg_val_loss = 0
        avg_val_acc = 0
        total_val = 0
        correct_val = 0

        ###################
        # train the model #
        ###################

        train_iters = int(training_u_amount/batch_size)
        # TQDM progress bar definition, for visualization purposes
        pbar_train = tqdm(enumerate(range(train_iters)),
                          total=train_iters,
                          unit=" iter",
                          leave=False,
                          file=sys.stdout,
                          desc='Train epoch ' + str(epoch + 1) + '/' + str(
                              epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (avg_train_loss, avg_val_acc))

        for i, ii in pbar_train:
            # for i, batch in enumerate(train_data_gen.flow()):
            # Epoch Training

            batch = next(train_data_gen)

            X = np.asarray(batch[0])
            p = np.argmax(np.asarray(batch[1]), axis=1)

            U = np.asarray(batch[2])
            q = np.asarray(batch[3])

            # Converting in PyTorch tensors
            X = torch.from_numpy(X.transpose((0, 3, 1, 2))).to(device, dtype=torch.float)
            p = torch.from_numpy(p).to(device, dtype=torch.long)
            U = torch.from_numpy(U.transpose((0, 3, 1, 2))).to(device, dtype=torch.float)
            q = torch.from_numpy(q).to(device, dtype=torch.float)

            model.train()
            # Forward pass: compute predicted y by passing x to the model.
            p_pred = model(X)
            q_pred = model(U)
            q_pred = torch.softmax(q_pred, dim=1)


            # Compute loss
            supervised_loss = supervised_criterion(p_pred, p)
            consistency_loss = consistency_criterion(q_pred, q)

            loss = supervised_loss + 75 * consistency_loss

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

            # Saving losses
            train_losses_item.append(loss.cpu().item())
            avg_train_loss = np.mean(train_losses_item)

            # Accuracy calculation
            _, predicted = torch.max(p_pred.data, 1)
            total_train += p.size(0)
            correct_train += (predicted == p).sum().item()
            train_acc_s_item.append(correct_train / total_train)
            avg_train_acc = np.mean(train_acc_s_item)

            # Update progress bar loss values
            pbar_train.set_description(
                'Train epoch ' + str(epoch + 1) + '/' + str(epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                avg_train_loss, avg_train_acc))

        # Saving train avg metrics
        avg_train_losses.append(avg_train_loss)
        avg_train_acc_s.append(avg_train_acc)

        ######################
        # validate the model #
        ######################

        model.eval()  # prep model for evaluation

        # TQDM progress bar definition, for visualization purposes
        val_iters = int(validation_amount/batch_size)
        pbar_val = tqdm(enumerate(range(val_iters)),
                        total=val_iters,
                        unit=" iter",
                        leave=False,
                        file=sys.stdout,
                        desc='Validation epoch ' + str(epoch + 1) + '/' + str(
                            epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (avg_val_loss, avg_val_acc))

        for i, batch in pbar_val:
            # Epoch validation

            X, p = next(val_data_gen)
            X = torch.from_numpy(np.asarray(X).transpose((0, 3, 1, 2))).to(device, dtype=torch.float)
            p = torch.from_numpy(np.asarray(p)).to(device, dtype=torch.long)

            # forward pass: compute predicted outputs by passing inputs to the model
            y_pred = model(X)
            # Compute and print loss.
            loss = supervised_criterion(y_pred, p)
            # record validation loss

            # Saving losses
            val_losses_item.append(loss.cpu().item())
            avg_val_loss = np.mean(val_losses_item)

            # Accuracy calculation
            _, predicted = torch.max(y_pred.data, 1)
            total_val += p.size(0)
            correct_val += (predicted == p).sum().item()
            val_acc_s_item.append(correct_val / total_val)
            avg_val_acc = np.mean(val_acc_s_item)
            val_losses_item.append(loss.item())

            # Update progress bar loss values
            pbar_val.set_description(
                'Validation epoch ' + str(epoch + 1) + '/' + str(epochs) + '   Loss: %.4f   Accuracy: %.3f  ' % (
                avg_val_loss, avg_val_acc))

        # Saving val avg metrics
        avg_val_losses.append(avg_val_loss)
        avg_val_acc_s.append(avg_val_acc)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_val_acc)
        checkpoint(avg_val_acc, model)

        # Reduce lr on plateau
        scheduler.step(avg_val_acc)

        if early_stopping.early_stop:
            print("Early stopping")
            break









    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    train_dl()



