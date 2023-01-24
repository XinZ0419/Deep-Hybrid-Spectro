import wandb
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(epoch, num_epochs, model, criterion, device, train_loader, optimizer, section):
    model.train()
    y_true = []
    y_pred = []
    running_loss = 0

    for i in train_loader:
        data_raw, data_freq, target, _, _ = i
        data_raw, data_freq, target = data_raw.to(device), data_freq.to(device), target.to(device)
        data_raw = torch.squeeze(data_raw)
        data_freq = torch.unsqueeze(data_freq, dim=1)

        # Forward
        if section == 'fft':
            output = model(data_freq.float())
        elif section == 'stft':
            output = model(data_freq.float())
        elif section == 'raw':
            output = model(data_raw.float())
        else:
            output = model(data_raw.float(), data_freq.float())

        loss = criterion(output, target.long().squeeze())
        running_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Predictions
        pred = np.round(output.cpu().detach())
        target = np.round(target.cpu().detach())
        y_pred.extend(pred.tolist())
        y_true.extend(target.tolist())

    y_pred = np.argmax(y_pred, axis=1)
    train_loss = running_loss / len(train_loader)

    # performance
    train_accuracy = accuracy_score(y_true, y_pred)
    train_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    train_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    train_f1 = f1_score(y_true, y_pred, average='macro')

    if (epoch + 1) % 100 == 0:
        print('In epoch {}/{}'.format(epoch + 1, num_epochs))
        print("[train metrics] loss:{:.4f} accuracy:{:.4f} recall:{:.4f} precision:{:.4f} f1:{:.4f}".format(train_loss,
                                                                                                            train_accuracy,
                                                                                                            train_recall,
                                                                                                            train_precision,
                                                                                                            train_f1))
    wandb.log({'training_loss': train_loss, 'train_accuracy': train_accuracy, 'train_recall': train_recall,
               'train_precision': train_precision, 'train_f1': train_f1, 'epoch': epoch})
    return train_loss, train_accuracy
