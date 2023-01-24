import sys
import torch
import wandb
import umap.umap_ as umap
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
from sklearn.model_selection import train_test_split

from modules.test import test
from modules.train import train
from modules.dataset import Classifier
from modules.validation import validation
from modules.model import LinearModel_fused_stftBased
from modules.utils import setup_seed, get_activation, visualization
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
parser.add_argument('--extra_signal_number', type=int, default=1 * 50 + 13,
                    help='count the multi files before running!e.g.(1*50 + 13) for female_1')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--drop_prob', type=float, default=0.3)
parser.add_argument('--train_proportion', type=float, default=0.6)
parser.add_argument('--train_val_proportion', type=float, default=0.5)

parser.add_argument('--section1', type=str, default='stft')
parser.add_argument('--section2', type=str, default='raw')
parser.add_argument('--section3', type=str, default='fused_stft')
parser.add_argument('--raw_data_file', type=str, default='example/1st sensor/motion speaking')
parser.add_argument('--save_ckpt_dir', type=str, default='example/1st sensor/visu_data/')
parser.add_argument('--processed_data_folder', type=str, default='example/1st sensor/processed_data/')
parser.add_argument('--baseline_model', type=str, default='example/1st sensor/result_stftBased_512/trained_models'
                                                          '/Model1_fused_stft_Fold_0_allclass_twoinputs_net_parameter.pth')
parser.add_argument('--processed_data_file', type=str,
                    default='example/1st sensor/processed_data/All_data_twoinputs.npy')

opt = parser.parse_args()
print(opt)


def visu():
    activation = {}
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

    all_data = Classifier(opt)
    num_samples = len(all_data.data_raw)
    print('the number of total samples is ', num_samples)
    all_data.data_raw = all_data.data_raw.reshape(all_data.data_raw.shape[0], 1, all_data.data_raw.shape[1])
    all_loader = dataloader.DataLoader(all_data, shuffle=False, batch_size=num_samples)

    # UMAP visualization
    model_comb_final.visu.register_forward_hook(get_activation('visu'))
    second_layer_output = visualization(model_comb_final, device, all_loader)

    mapper = umap.UMAP(random_state=opt.seed).fit_transform(second_layer_output.squeeze().cpu())

    plt.figure(figsize=(10, 10))
    plt.scatter(mapper[:, 0], mapper[:, 1], c=all_data.label.squeeze(), s=10, cmap='Spectral')
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.savefig(opt.save_ckpt_dir + 'UMAP_All-class.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    wandb.init(config=opt, project='Sensor_stftBased', entity='xinz', name='visu_BaselineData')
    setup_seed(opt.seed)

    visu()
