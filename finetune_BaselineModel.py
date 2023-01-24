import sys
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from sklearn.model_selection import train_test_split

from modules.test import test
from modules.train import train
from modules.dataset import Classifier
from modules.validation import validation
from modules.model import LinearModel_fused_stftBased
from modules.utils import setup_seed, plot_loss, plot_acc
from modules.data_processing_utils import read_files, cut_and_class, filter_and_fft

sys.path.append("..")

parser = argparse.ArgumentParser(description='Sensor')

parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--partic', type=bool, default=False)
parser.add_argument('--remove_dc', type=bool, default=True)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=11)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--num_samples_to_keep_after', type=int, default=1500)
parser.add_argument('--num_samples_to_keep_before', type=int, default=1500)
parser.add_argument('--sampling_rate', type=int, default=1000, help='with unit in HZ')
parser.add_argument('--length_keep', type=int, default=3500, help='raw data length at least')
parser.add_argument('--num_samples_to_keep', type=int, default=3000, help='raw data length to keep')
parser.add_argument('--num_participate', type=int, default=1, help='how many participates in the folder?')
parser.add_argument('--extra_signal_number', type=int, default=1*50 + 13, help='count the multi files before running!e.g.(1*50 + 13) for female_1')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--train_proportion', type=float, default=0.6)
parser.add_argument('--train_val_proportion', type=float, default=0.5)

parser.add_argument('--section1', type=str, default='stft')
parser.add_argument('--section2', type=str, default='raw')
parser.add_argument('--section3', type=str, default='fused_stft')
parser.add_argument('--raw_data_file', type=str, default='example/male_0_p3/motion speaking')
parser.add_argument('--save_ckpt_dir', type=str, default='example/male_0_p3/result_finetune/')
parser.add_argument('--processed_data_folder', type=str, default='example/male_0_p3/processed_data/')
parser.add_argument('--baseline_model', type=str, default='example/baseline/result_stftBased_512/trained_models'
                                                          '/Model1_fused_stft_Fold_0_allclass_twoinputs_net_parameter.pth')
parser.add_argument('--processed_data_file', type=str, default='example/male_0_p3/processed_data/All_data_twoinputs_male_0_p3.npy')

opt = parser.parse_args()
print(opt)


def finetune():
    fold = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load & freeze baseline model
    model_comb_final = LinearModel_fused_stftBased(opt.num_samples_to_keep, opt.num_classes).to(device)
    state_dict = torch.load(opt.baseline_model)
    model_comb_final.load_state_dict(state_dict)
    count = 0
    for child in model_comb_final.children():
        count += 1
        if count < 3:
            for param in child.parameters():
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_comb_final.parameters()), lr=opt.lr)

    all_data = Classifier(opt)
    num_samples = len(all_data.data_raw)
    all_data.data_raw = all_data.data_raw.reshape(all_data.data_raw.shape[0], 1, all_data.data_raw.shape[1])
    train_val_data, test_data = train_test_split(all_data, train_size=opt.train_val_proportion, random_state=opt.seed, shuffle=True, stratify=all_data.label)
    train_data, val_data = train_test_split(all_data, train_size=opt.train_proportion, random_state=opt.seed, shuffle=True, stratify=all_data.label)
    train_loader = dataloader.DataLoader(train_data, batch_size=opt.train_batch_size)
    val_loader = dataloader.DataLoader(val_data, shuffle=False, batch_size=1)
    test_loader = dataloader.DataLoader(test_data, shuffle=False, batch_size=1)
    all_loader = dataloader.DataLoader(all_data, shuffle=False, batch_size=1)

    # # test directly
    # test_acc_directly = test(opt, model_comb_final, device, all_loader, opt.section3, fold, opt.save_ckpt_dir, person_based=opt.partic)

    # finetune
    train_losses = []
    train_acc = []
    eval_losses = []
    eval_acc = []
    min_loss = 100

    for epoch in range(opt.num_epochs):
        print('In epoch {}/{}'.format(epoch + 1, opt.num_epochs))
        train_loss, train_accuracy = train(epoch, opt.num_epochs, model_comb_final, criterion, device, train_loader, optimizer, opt.section3)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)

        val_loss, val_accuracy = validation(epoch, model_comb_final, criterion, device, val_loader, opt.section3)
        eval_losses.append(val_loss)
        eval_acc.append(val_accuracy)

        if val_loss < min_loss:
            min_loss = val_loss
            print(f'save model_{fold}====in epoch {epoch + 1}/{opt.num_epochs}')
            torch.save(model_comb_final.state_dict(), f'{opt.save_ckpt_dir}trained_models/Model1_{opt.section3}_Fold_{fold}_allclass_twoinputs_net_parameter.pth')

    plot_loss(train_losses, eval_losses, opt.save_ckpt_dir, opt.section3, fold)
    plot_acc(train_acc, eval_acc, opt.save_ckpt_dir, opt.section3, fold)

    test_acc_finetune = test(opt, model_comb_final, device, test_loader, opt.section3, fold, opt.save_ckpt_dir, person_based=opt.partic)


if __name__ == '__main__':
    wandb.init(config=opt, project='Sensor_stftBased_fintune', entity='xinz', name='male_0_p3_finetune')
    setup_seed(opt.seed)

    file_list, all_data, _ = read_files(opt)
    single_signals, single_time, new_file_list, all_classes, partic_all, pc_all = cut_and_class(opt, all_data,
                                                                                                file_list)
    final_array = filter_and_fft(opt, single_signals, single_time, all_classes, partic_all, pc_all, new_file_list)
    with open(opt.processed_data_file, 'wb') as f:
        np.save(f, final_array)
    print("Processed data is saved to:", opt.processed_data_file)

    listes = pd.DataFrame(new_file_list)
    classes = pd.DataFrame(all_classes)
    partic = pd.DataFrame(partic_all)
    pc = pd.DataFrame(pc_all)
    result = pd.concat([listes, classes, partic, pc], axis=1)
    result.to_csv(opt.processed_data_folder + 'compare_classes.csv')

    finetune()
