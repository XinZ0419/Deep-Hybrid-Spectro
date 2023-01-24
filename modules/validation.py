import wandb
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def validation(epoch, model, criterion, device, validation_loader, section):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0

    with torch.no_grad():
        for i in validation_loader:
            data_raw, data_freq, target, _, _ = i
            data_raw, data_freq, target = data_raw.to(device), data_freq.to(device), target.to(device)
            data_raw = torch.squeeze(data_raw)
            # data_fft = torch.squeeze(data_fft)
            data_freq = torch.unsqueeze(data_freq, dim=1)

            if section == 'fft':
                output = model(data_freq.unsqueeze(0).float())
            elif section == 'stft':
                output = model(data_freq.float())
            elif section == 'raw':
                output = model(data_raw.unsqueeze(0).float())
            else:
                output = model(data_raw.unsqueeze(0).float(), data_freq.float())

            loss = criterion(torch.squeeze(output), torch.squeeze(target.long()))
            running_loss += loss.item()

            pred = np.round(output.cpu())
            target = target.float()
            y_true.extend(target.cpu().tolist())
            y_pred.extend(pred.tolist())

    y_pred = np.argmax(y_pred, axis=1)
    val_loss = running_loss / len(validation_loader)

    # performance
    val_accuracy = accuracy_score(y_true, y_pred)
    val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    val_f1 = f1_score(y_true, y_pred, average='macro')

    if (epoch + 1) % 100 == 0:
        print("[valid  metrics] loss:{:.4f} accuracy:{:.4f} recall:{:.4f} precision:{:.4f} f1:{:.4f}".format(val_loss,
                                                                                                         val_accuracy,
                                                                                                         val_recall,
                                                                                                         val_precision,
                                                                                                         val_f1))
    wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_recall': val_recall,
               'val_precision': val_precision, 'val_f1': val_f1, 'epoch': epoch})

    return val_loss, val_accuracy
