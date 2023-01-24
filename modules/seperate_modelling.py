import copy
import torch
import torch.nn as nn

from modules.test import test
from modules.train import train
from modules.validation import validation
from modules.utils import reset_weights, plot_loss, plot_acc
from modules.model import LinearModel_fft, LinearModel_stft, LinearModel_raw, LinearModel_fused, LinearModel_fused_stftBased


def sep_mod_fold(opt, fold, section, device, train_loader, val_loader, test_loader, model_freq=None, model_raw=None):
    if section == 'fft':
        model = LinearModel_fft(opt.num_samples_to_keep, opt.num_classes, opt.drop_prob).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    elif section == 'stft':
        model = LinearModel_stft(opt.num_classes, drop_prob=opt.drop_prob).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    elif section == 'raw':
        model = LinearModel_raw(opt.num_samples_to_keep, opt.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    elif section == 'fused':
        # load =====================================================================
        model = LinearModel_fused(opt.num_samples_to_keep, opt.num_samples_to_keep, opt.num_classes, opt.drop_prob).to(device)
        model_dict = model.state_dict()
        fft_dict = model_freq.state_dict()
        raw_dict = model_raw.state_dict()
        fft_dict = {k: v for k, v in fft_dict.items() if k in model_dict}
        raw_dict = {k: v for k, v in raw_dict.items() if k in model_dict}
        model_dict.update(fft_dict)
        model_dict.update(raw_dict)
        model.load_state_dict(model_dict)
        # freeze ===================================================================
        count = 0
        for child in model.children():
            count += 1
            if count < 3:
                for param in child.parameters():
                    param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    elif section == 'fused_stft':
        # load =====================================================================
        model = LinearModel_fused_stftBased(opt.num_samples_to_keep, opt.num_classes, drop_prob=opt.drop_prob).to(device)
        model_dict = model.state_dict()
        stft_dict = model_freq.state_dict()
        raw_dict = model_raw.state_dict()
        stft_dict = {k: v for k, v in stft_dict.items() if k in model_dict}
        raw_dict = {k: v for k, v in raw_dict.items() if k in model_dict}
        model_dict.update(stft_dict)
        model_dict.update(raw_dict)
        model.load_state_dict(model_dict)
        # freeze ===================================================================
        count = 0
        for child in model.children():
            count += 1
            if count < 3:
                for param in child.parameters():
                    param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    else:
        raise TypeError('Wrong section type!')

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    eval_losses = []
    train_acc = []
    eval_acc = []
    test_max_acc = 0.0

    model.apply(reset_weights)  # reset the model
    min_loss = 100              # random big number

    for epoch in range(opt.num_epochs):

        train_loss, train_accuracy = train(epoch, opt.num_epochs, model, criterion, device, train_loader, optimizer, section)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)

        val_loss, val_accuracy = validation(epoch, model, criterion, device, val_loader, section)
        eval_losses.append(val_loss)
        eval_acc.append(val_accuracy)

        if val_loss < min_loss:
            min_loss = val_loss
            print(f'save model_{fold}====in epoch {epoch + 1}/{opt.num_epochs}')
            torch.save(model.state_dict(), f'{opt.save_ckpt_dir}trained_models/Model1_{section}_Fold_{fold}_allclass_twoinputs_net_parameter.pth')

    # plot training & testing loss
    plot_loss(train_losses, eval_losses, opt.save_ckpt_dir, section, fold)

    # plot training & testing accuracy
    plot_acc(train_acc, eval_acc, opt.save_ckpt_dir, section, fold)

    test_acc = test(opt, model, device, test_loader, section, fold, opt.save_ckpt_dir, person_based=opt.partic)
    if test_acc > test_max_acc:
        test_max_acc == test_acc
        model_final = copy.deepcopy(model)

    return model_final


if __name__ == '__main__':
    section = 'fft'
    sep_mod_fold(section)
