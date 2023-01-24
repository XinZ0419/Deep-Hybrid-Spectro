import torch
import random
import itertools
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
            data_raw, data_freq, target, _, _ = i
            data_raw, data_freq, target = data_raw.to(device), data_freq.to(device), target.to(device)
            data_raw = torch.squeeze(data_raw)
            data_freq = torch.unsqueeze(data_freq, dim=1)

            # the model on the data
            output = model(data_raw.float(), data_freq.float())
    return activation['visu']


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def get_confusion_matrix(trues, preds):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    conf_matrix = confusion_matrix(trues, preds, labels=labels)
    return conf_matrix


def plot_confusion_matrix(cm, dir, section, fold, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.savefig(f'{dir}/plots/confusion_matrix_{section}_fold{str(fold)}.png')
    plt.show()


def plot_single_confusion_matrix(cm, dir, section, fold, classes, partic, normalize=False, title='Confusion matrix',
                                 cmap=plt.cm.Blues):
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
    plt.savefig(f'{dir}/plots/confusion_matrix_{section}_fold{str(fold)}_partic{str(partic)}.png')
    plt.show()


font = {'size': 13, 'weight': 'normal', 'color': 'black', 'style': 'normal'}


def get_roc_auc(trues, preds, dir, section, fold):
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
    plt.savefig(f'{dir}/plots/ROC_All-class_{section}_fold{str(fold)}.png')
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
    all_df.to_csv(f'{dir}/plots/roc_data_all_class_{section}_fold{str(fold)}.csv')


def plot_loss(train_loss, eval_loss, dir=None, section=None, fold=None):
    plt.plot(train_loss)
    plt.plot(eval_loss)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig(f'{dir}/plots/losses_All-class-{section}-fold{str(fold)}.png')
    plt.show()


def plot_acc(train_acc, eval_acc, dir=None, section=None, fold=None):
    plt.plot(train_acc)
    plt.plot(eval_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.savefig(f'{dir}/plots/Accuracy_All-class-{section}-fold{str(fold)}.png')
    plt.show()
