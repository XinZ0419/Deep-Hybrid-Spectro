import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import itertools

from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from modules.dataset import Classifier
from modules.model import LinearModel_fft, LinearModel_raw, LinearModel_fused


def separate_modelling(part):
    debug = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    start_from_checkpoint = False

    train_proportion = 0.8
    val_prop = 0.3  # Proportion from the train

    input_size = 3000
    num_classes = 11
    num_epochs = 1300
    batch_size = 10
    learning_rate = 1e-6

    # KFold validation
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)

    save_dir = 'Models/Model_v1'
    file_name = 'D://Xin Zhang/Sensor/conbination/processed_data/All_data_twoinputs.npy'

    all_data = Classifier(file_name)
    num_samples = len(all_data.data_raw)
    all_data.data_raw = all_data.data_raw.reshape(all_data.data_raw.shape[0], 1, all_data.data_raw.shape[1])
    all_data.data_fft = all_data.data_fft.reshape(all_data.data_fft.shape[0], 1, all_data.data_fft.shape[1])

    n_train = round(num_samples * train_proportion)
    train_data, test_data = torch.utils.data.random_split(all_data, [n_train, num_samples - n_train])
    test_loader = dataloader.DataLoader(test_data, shuffle=False, batch_size=batch_size)
    all_loader = dataloader.DataLoader(all_data, shuffle=False, batch_size=num_samples)

    # Creating model and setting loss and optimizer
    if section == 'fft':
        model = LinearModel_fft(input_size, num_classes).to(device)
    else:
        model = LinearModel_raw(input_size,num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(n_train))):
        print('Fold {}'.format(fold + 1))
        train_losses = []
        eval_losses = []
        train_acc = []
        eval_acc = []

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = dataloader.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        val_loader = dataloader.DataLoader(train_data, batch_size=batch_size, sampler=val_sampler)

        # reset the model
        model.apply(reset_weights)
        min_loss = 100  # ramdom big number

        for epoch in range(num_epochs):

            print('In epoch {}/{}'.format(epoch + 1, num_epochs))
            train_loss, train_accuracy = train(model, criterion, device, train_loader, optimizer, part)
            train_losses.append(train_loss)
            train_acc.append(train_accuracy)

            val_loss, val_accuracy = validation(model, criterion, device, val_loader, part)
            eval_losses.append(val_loss)
            eval_acc.append(val_accuracy)

            if val_loss < min_loss:
                min_loss = val_loss
                print(f'================================save model_{fold + 1}================================')
                # torch.save(model.state_dict(), f'Model1_fft_Fold_{fold+1}_allclass_twoinputs_net_parameter.pth')

        # plot training & testing loss
        plt.plot(train_losses)
        plt.plot(eval_losses)
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train', 'Valid'])
        plt.title('Train vs Valid Losses')
        plt.show()

        # plot training & testing accuracy
        plt.plot(train_acc)
        plt.plot(eval_acc)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train', 'Valid'])
        plt.title('Train vs Valid Accuracy')
        plt.savefig("Accuracy_All-class.png")
        plt.show()


def train(model, criterion, device, train_loader, optimizer, section):
    model.train()
    y_true = []
    y_pred = []
    running_loss = 0

    for i in train_loader:
        data_raw, data_fft, target = i
        data_raw, data_fft, target = data_raw.to(device), data_fft.to(device), target.to(device)
        data_raw = torch.squeeze(data_raw)
        data_fft = torch.squeeze(data_fft)

        # Forward
        if section == 'fft':
            output = model(data_fft.float())
        elif section == 'raw':
            output = model(data_raw.float())
        else:
            output = model(data_raw.float(), data_fft.float())

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
    train_recall = recall_score(y_true, y_pred, average='macro')
    train_precision = precision_score(y_true, y_pred, average='macro')
    train_f1 = f1_score(y_true, y_pred, average='macro')

    print("[train metrics] loss:{:.4f} accuracy:{:.4f} recall:{:.4f} precision:{:.4f} f1:{:.4f}".format(train_loss,
                                                                                                        train_accuracy,
                                                                                                        train_recall,
                                                                                                        train_precision,
                                                                                                        train_f1))
    return train_loss, train_accuracy


def validation(model, criterion, device, validation_loader, section):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0

    with torch.no_grad():
        for i in validation_loader:
            data_raw, data_fft, target = i
            data_raw, data_fft, target = data_raw.to(device), data_fft.to(device), target.to(device)
            data_raw = torch.squeeze(data_raw)
            data_fft = torch.squeeze(data_fft)

            if section == 'fft':
                output = model(data_fft.float())
            elif section == 'raw':
                output = model(data_raw.float())
            else:
                output = model(data_raw.float(), data_fft.float())

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
    val_recall = recall_score(y_true, y_pred, average='macro')
    val_precision = precision_score(y_true, y_pred, average='macro')
    val_f1 = f1_score(y_true, y_pred, average='macro')

    print("[valid  metrics] loss:{:.4f} accuracy:{:.4f} recall:{:.4f} precision:{:.4f} f1:{:.4f}".format(val_loss,
                                                                                                         val_accuracy,
                                                                                                         val_recall,
                                                                                                         val_precision,
                                                                                                         val_f1))

    return val_loss, val_accuracy


def reset_weights(m):
    for layer in m.children():
        for param in layer.parameters():
            if param.requires_grad == True:
                flag = True
            else:
                flag = False
        if hasattr(layer, 'reset_parameters') & flag:
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def visualization(model, device, all_loader):
    model.eval()
    with torch.no_grad():
        for i in all_loader:
            data_raw, data_fft, target = i
            data_raw, data_fft, target = data_raw.to(device), data_fft.to(device), target.to(device)
            data_raw = torch.squeeze(data_raw)
            data_fft = torch.squeeze(data_fft)

            # the model on the data
            output = model(data_raw.float(), data_fft.float())
    return activation['visu']


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def get_confusion_matrix(trues, preds):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    conf_matrix = confusion_matrix(trues, preds, labels)
    return conf_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar(fraction=0.03, pad=0.06)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt), verticalalignment="center", horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('Actual Throat Activities',fontsize=13)
    # plt.xlabel('Predicted Throat Activities',fontsize=13)
    plt.savefig('confusion_matrix.png')
    plt.show()


font = {'size': 13, 'weight': 'normal', 'color': 'black', 'style': 'normal'}


def get_roc_auc(trues, preds):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(trues, preds)

    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.xlabel('1 - specificity', fontdict=font, fontweight='bold')
    plt.ylabel('Sensitivity', fontdict=font, fontweight='bold')
    #     plt.title('ROC-AUC for all-class',fontdict=font)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average', color='deeppink', linestyle=':')
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average', color='navy', linestyle=':')
    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'green', 'gold', 'black', 'pink', 'grey', 'orange'])

    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='{0}'.format(i + 1))
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='class {0} (AUC={1:0.2f})'.format(i, roc_auc[i]))
    #     plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.legend(loc="lower right", fontsize=13, frameon=False, ncol=2)
    plt.savefig("ROC_All-class.png")
    plt.show()

    # write roc data into csv file
    all_df = pd.DataFrame()
    for i in range(nb_classes):
        fpr_temp_df = pd.DataFrame(fpr[i])
        all_df = pd.concat([all_df, fpr_temp_df], axis=1)
        tpr_temp_df = pd.DataFrame(tpr[i])
        all_df = pd.concat([all_df, tpr_temp_df], axis=1)
    macro_fpr_temp_df = pd.DataFrame(fpr['macro'])
    all_df = pd.concat([all_df, macro_fpr_temp_df], axis=1)
    macro_tpr_temp_df = pd.DataFrame(tpr['macro'])
    all_df = pd.concat([all_df, macro_tpr_temp_df], axis=1)
    micro_fpr_temp_df = pd.DataFrame(fpr['micro'])
    all_df = pd.concat([all_df, micro_fpr_temp_df], axis=1)
    micro_tpr_temp_df = pd.DataFrame(tpr['micro'])
    all_df = pd.concat([all_df, micro_tpr_temp_df], axis=1)

    p_col = ['0 fpr', '0 tpr', '1 fpr', '1 tpr', '2 fpr', '2 tpr', '3 fpr', '3 tpr', '4 fpr', '4 tpr', '5 fpr', '5 tpr',
             '6 fpr', '6 tpr', '7 fpr', '7 tpr', '8 fpr', '8 tpr', '9 fpr', '9 tpr', '10 fpr', '10 tpr', 'macro fpr',
             'macro tpr', 'micro fpr', 'micro tpr']
    all_df.columns = p_col
    all_df.to_csv('roc_data_all_class.csv')


if __name__ == '__main__':
    section = 'fft'
    separate_modelling(section)
