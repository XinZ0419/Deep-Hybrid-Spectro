import sys
import torch
import wandb
import argparse
import numpy as np
import pandas as pd

import torch.utils.data.dataloader as dataloader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from modules.utils import setup_seed
from modules.dataset import Classifier
from modules.seperate_modelling import sep_mod_fold
from modules.data_processing_utils import read_files, cut_and_class, filter_and_fft

sys.path.append("..")

parser = argparse.ArgumentParser(description='Sensor')

parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--partic', type=bool, default=True)
parser.add_argument('--remove_dc', type=bool, default=True)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_classes', type=int, default=11)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--num_samples_to_keep_after', type=int, default=1500)
parser.add_argument('--num_samples_to_keep_before', type=int, default=1500)
parser.add_argument('--sampling_rate', type=int, default=1000, help='with unit in HZ')
parser.add_argument('--length_keep', type=int, default=3500, help='raw data length at least')
parser.add_argument('--num_samples_to_keep', type=int, default=3000, help='raw data length to keep')
parser.add_argument('--num_participate', type=int, default=2, help='how many participates in the folder?')
parser.add_argument('--extra_signal_number', type=int, default=3*201 + 2 + 3*230 + (13 + 23 + 25 + 29 + 47)*2, help='count the multi files before running!')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--train_proportion', type=float, default=0.8)

parser.add_argument('--section1', type=str, default='stft')
parser.add_argument('--section2', type=str, default='raw')
parser.add_argument('--section3', type=str, default='fused_stft')
parser.add_argument('--raw_data_file', type=str, default='example/baseline/motion speaking')
parser.add_argument('--save_ckpt_dir', type=str, default='example/baseline/result_stftBased_512/')
parser.add_argument('--processed_data_folder', type=str, default='example/baseline/processed_data/')
parser.add_argument('--processed_data_file', type=str, default='example/baseline/processed_data/All_data_twoinputs.npy')

opt = parser.parse_args()
print(opt)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_data = Classifier(opt)
    num_samples = len(all_data.data_raw)
    all_data.data_raw = all_data.data_raw.reshape(all_data.data_raw.shape[0], 1, all_data.data_raw.shape[1])
    n_train = round(num_samples * opt.train_proportion)

    train_data, test_data = train_test_split(all_data, train_size=opt.train_proportion, random_state=opt.seed, shuffle=True, stratify=all_data.pl)
    test_loader = dataloader.DataLoader(test_data, shuffle=False, batch_size=1)
    all_loader = dataloader.DataLoader(all_data, shuffle=False, batch_size=1)

    train_pl_list = []
    for i in train_data:
        train_pl_list.append(i[-1])

    # KFold validation
    k = 5
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=opt.seed)
    for fold, (train_idx, val_idx) in enumerate(splits.split(train_data, train_pl_list)):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = dataloader.DataLoader(train_data, batch_size=opt.train_batch_size, sampler=train_sampler)
        val_loader = dataloader.DataLoader(train_data, batch_size=1, sampler=val_sampler)

        model_fft_final = sep_mod_fold(opt, fold, opt.section1, device, train_loader, val_loader, test_loader)
        model_raw_final = sep_mod_fold(opt, fold, opt.section2, device, train_loader, val_loader, test_loader)

        model_fused_final = sep_mod_fold(opt, fold, opt.section3, device, train_loader, val_loader, test_loader, model_fft_final, model_raw_final)


if __name__ == '__main__':
    wandb.init(config=opt, project='Sensor_stftBased_5fold', entity='xinz', name='opt1')
    setup_seed(opt.seed)

    file_list, all_data, _ = read_files(opt)
    single_signals, single_time, new_file_list, all_classes, partic_all, pc_all = cut_and_class(opt, all_data, file_list)
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

    main()
