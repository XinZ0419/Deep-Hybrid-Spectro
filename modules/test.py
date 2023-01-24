import wandb
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from modules.utils import get_confusion_matrix, plot_confusion_matrix, plot_single_confusion_matrix, get_roc_auc


def test(opt, model, device, test_loader, section, fold, save_dir, person_based=False):
    # model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    p_true = []
    y_pred = []
    test_acc = []
    y_pred_prob = []

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            data_raw, data_freq, target, partic, _ = i
            data_raw, data_freq, target, partic = data_raw.to(device), data_freq.to(device), target.to(device), partic.to(
                device)
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

            # Predictions
            pred = np.round(output.cpu())
            target = target.float()
            y_true.extend(target.cpu().tolist())
            p_true.extend(partic.cpu().tolist())
            y_pred_prob.extend(output.tolist())
            y_pred.extend(pred.tolist())

    y_pred = np.argmax(y_pred, axis=1)

    df_y_true = pd.DataFrame(y_true, columns=['y_true'])
    df_p_true = pd.DataFrame(p_true, columns=['participant'])
    df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    result = pd.concat([df_p_true, df_y_true, df_y_pred], axis=1)
    result.to_csv(f'{save_dir}/plots/predict_info_{section}_fold{str(fold)}.csv')

    # performance
    test_accuracy = accuracy_score(y_true, y_pred)
    test_acc.append(test_accuracy)
    test_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_true, y_pred, average='macro')

    print("[test  metrics] accuracy:{:.4f} recall:{:.4f} precision:{:.4f} f1:{:.4f}".format(test_accuracy,
                                                                                            test_recall,
                                                                                            test_precision,
                                                                                            test_f1))
    wandb.log({'test_accuracy': test_accuracy, 'test_recall': test_recall,
               'test_precision': test_precision, 'test_f1': test_f1})
    print("*****************************************************************************************")

    # confusion matrix
    print(classification_report(y_true, y_pred))
    conf_matrix = get_confusion_matrix(y_true, y_pred)
    attack_types = ['Yes', 'No', 'One', 'Two', 'Shake', 'Nod', 'Stretch', 'Nod+No', 'Nod+Yes', 'Shake+No', 'Shake+Yes']
    plot_confusion_matrix(conf_matrix, save_dir, section, fold, classes=attack_types, normalize=True,
                          title='Normalized confusion matrix')

    # confusion matrix for each one
    if person_based:
        save_dir_single = save_dir + 'single/'
        for i in range(opt.num_participate):
            temp = result[result['participant'] == i+1]
            temp_y_true = temp['y_true']
            temp_y_pred = temp['y_pred']
            conf_matrix = get_confusion_matrix(temp_y_true, temp_y_pred)
            plot_single_confusion_matrix(conf_matrix, save_dir_single, section, fold, classes=attack_types, partic=i+1, normalize=False,
                                  title='Normalized confusion matrix')

    # ROC curve
    y_true_binary = label_binarize(y_true, classes=[i for i in range(11)])
    y_pred_prob = np.array(y_pred_prob)
    get_roc_auc(y_true_binary, y_pred_prob, save_dir, section, fold)

    return test_accuracy
