import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize,RandomHorizontalFlip,RandomCrop
from tqdm.notebook import tqdm
import torchshow as ts
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from ASSET.models import *
from ASSET.new_poi_util import *
from CTRL.methods import set_model
from CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from CTRL.utils.frequency import PoisonFre
from DRUPE.models.simclr_model import SimCLR
from DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from DECREE.imagenet import getBackdoorImageNet, get_processing
from DECREE.models import get_encoder_architecture_usage
from BadCLIP.pkgs.openai.clip import load as load_model
from utils import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature, MAE_test, MAE_error
import utils
from decoder_model import DecoderModel

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


def sift_dataset(backdoored_encoder, model, dataset, threshold):
    _, errors = MAE_error(backdoored_encoder, model, dataset, save_cuda = True)
    cln_idxs = np.where(np.array(errors) < threshold)[0]
    poi_idxs = np.where(np.array(errors) > threshold)[0]
    cln_dataset = torch.utils.data.Subset(dataset, cln_idxs)
    downstrm_test_backdoor_dataloader = DataLoader(cln_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    feature_bank_backdoor, label_bank_backdoor = predict_feature(backdoored_encoder, downstrm_test_backdoor_dataloader)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)
    print("After sifting, the number kept/the number sifted:", len(cln_idxs), len(poi_idxs))
    return nn_backdoor_loader, (len(cln_idxs), len(poi_idxs))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu the code runs on')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--poison_rate', default=0.01, type=float, help='')

    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_mask_ratio', default=0.99, type=float, help='mask ratio for decoder in the detection time')

    args = parser.parse_args()


    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ### load victim encoder
    if args.attack_type == 'badencoder':
        encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t12/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t12/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'ctrl':
        with open('./CTRL/args.pkl', 'rb') as handle:
            ctrl_args = pickle.load(handle)
        ctrl_args.data_path = './data/'
        ctrl_args.threat_model = 'our'
        print("load attack args:", ctrl_args)
        vic_model = set_model(ctrl_args).cuda()
        encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-0/' + 'epoch_81.pth.tar' # + 'last.pth.tar' #  + 'epoch_81.pth.tar'
        checkpoint = torch.load(encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'clip':
        with open('./DECREE/clip_text.pkl', 'rb') as handle:
            clip_args = pickle.load(handle)
        clip_args.pretrained_encoder = f'./DECREE/output/CLIP_text/cifar10_backdoored_encoder/model_69clip_text_atk0.05_41.pth'
        vic_model = get_encoder_architecture_usage(clip_args).cuda()
        checkpoint = torch.load(clip_args.pretrained_encoder, map_location='cpu', weights_only=True)
        vic_model.visual.load_state_dict(checkpoint['state_dict'])
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP' # assert
        args.image_size = 224 # assert
        print("load backdoor model from", clip_args.pretrained_encoder)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        checkpoint = torch.load('./BadCLIP/logs/nodefence_ours_final/checkpoints/epoch_10.pt', map_location='cpu', weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())):
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP'  # assert
        args.image_size = 224  # assert
        print("load backdoor model for BadCLIP")
    else:
        print("invalid mode")
        1/0

    ### prepare train dataset, test dataset
    if args.attack_type == 'badencoder' or args.attack_type == 'drupe':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz'  # depending on downstream tasks
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'
        shadow_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_clean = utils.CIFAR10_BACKDOOR(root='./data', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR(root='./data', train=False, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=1.0, lb_flag='backdoor')
    elif args.attack_type == 'ctrl':
        if args.poison_rate == 0:
            ctrl_args.poison_ratio = 0.1
            train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(ctrl_args)
            shadow_data = memory_loader.dataset
            poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size, ctrl_args.trigger_position, False, True)
            poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, ctrl_args.magnitude)
            test_loader = poison.test_loader
            test_pos_loader = poison.test_pos_loader
        else:
            ctrl_args.poison_ratio = args.poison_rate
            train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(ctrl_args)
            poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size, ctrl_args.trigger_position, False, True)
            poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, ctrl_args.magnitude)
            shadow_data = poison.train_pos_loader.dataset
            test_loader = poison.test_loader
            test_pos_loader = poison.test_pos_loader

        test_data_clean = test_loader.dataset
        test_data_backdoor = test_pos_loader.dataset
        memory_data = memory_loader.dataset

        shadow_data = utils.DummyDataset(shadow_data, transform=utils.test_transform)
        memory_data = utils.DummyDataset(memory_data, transform=utils.test_transform)
        test_data_clean = utils.DummyDataset(test_data_clean, transform=utils.test_transform)
        test_data_backdoor = utils.DummyDataset(test_data_backdoor, transform=utils.test_transform)
    elif args.attack_type == 'clip':
        trigger_file = './DECREE/trigger/' + 'trigger_pt_white_185_24.npz'
        shadow_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
                                             test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
                                               test_transform=utils.test_transform224, poison_rate=0, lb_flag='')

        test_data_clean = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,test_transform=utils.test_transform224, poison_rate=1, lb_flag='backdoor')

    elif args.attack_type == 'badclip':
        shadow_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                                     test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                               test_transform=utils.test_transform224, poison_rate=0, lb_flag='')

        test_data_clean = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                   test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                      test_transform=utils.test_transform224, poison_rate=1, lb_flag='backdoor')

    else:
        print("invalid dataset")
        1/0

    print("size of shadow_data", len(shadow_data))
    print("size of test backdoor/clean", len(test_data_backdoor), len(test_data_clean))
    downstrm_train_dataloader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                           pin_memory=True, drop_last=False)
    downstrm_test_backdoor_dataloader = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=16, pin_memory=True, drop_last=False)
    downstrm_test_clean_dataloader = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False,
                                                num_workers=16, pin_memory=True, drop_last=False)
    feature_bank_training, label_bank_training = predict_feature(backdoored_encoder, downstrm_train_dataloader)
    feature_bank_testing, label_bank_testing = predict_feature(backdoored_encoder, downstrm_test_clean_dataloader)
    feature_bank_backdoor, label_bank_backdoor = predict_feature(backdoored_encoder,  downstrm_test_backdoor_dataloader)
    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)


    # main loop - test
    result_record = {"ca_baseline":[], "asr_baseline":[], "ca_def":[], "asr_def":[]}
    input_size = feature_bank_training.shape[1]
    print('input_size', input_size)
    criterion = nn.CrossEntropyLoss()
    net = NeuralNet(input_size, [512, 256], 10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)

    epochs = args.epochs
    ba_acc, asr_acc = [], []

    for epoch in range(epochs):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)

        acc1 = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        acc2 = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
        ba_acc.append(acc1)
        asr_acc.append(acc2)

    print("BA best:", max(ba_acc), "ASR best:", max(asr_acc))
    result_record['ca_baseline'].append(ba_acc)
    result_record['asr_baseline'].append(asr_acc)
    with open('./plot_result/downstm_att{}_pr{}.pkl'.format(args.attack_type, args.poison_rate), 'wb') as f:  # open a text file
        pickle.dump(result_record, f)  # serialize the list

    ### Append to the algorithm removal
    # load decoder
    if args.attack_type == 'badencoder':
        tag = 'badencoder_mask9_patch4_id' # args.tag
    elif args.attack_type == 'drupe':
        tag = 'drupe_mask9_patch4_id'  # args.tag
    elif args.attack_type == 'ctrl':
        tag = 'ctrl_mask75_patch2_id_p2_flat'  # args.tag
    elif args.attack_type == 'clip':
        tag = 'clip_mask9_patch4_id'  # args.tag
    elif args.attack_type == 'badclip':
        tag = 'badclip_mask75_patch32_ood'  # args.tag

    result_dir = './DEDE_results/' + tag
    with open(result_dir + '/args.pkl', 'rb') as handle:
        dec_args = pickle.load(handle)
    dec_args.mask_ratio = args.test_mask_ratio  # test time

    if args.attack_type == 'clip' or args.attack_type == 'badclip':
        print("assert clip")
        dec_args.arch = 'CLIP'  # assert
        dec_args.image_size = 224  # assert
        model = DecoderModel(image_size=dec_args.image_size,
                             patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                             encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer,
                             arch=dec_args.arch).cuda()
    else:
        model = DecoderModel(patch_size=dec_args.patch_size, mask_ratio=dec_args.mask_ratio,
                         encoder_layer=dec_args.encoder_layer, decoder_layer=dec_args.decoder_layer).cuda()

    checkpoint = torch.load(result_dir + '/final.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("decoder model loaded from", result_dir)

    # testing
    if args.attack_type == 'clip' or args.attack_type == 'badclip':
        error_record = []
        _, errors = MAE_error(backdoored_encoder, model, test_data_clean, save_cuda = True)
        error_record.append(errors)
        _, errors = MAE_error(backdoored_encoder, model, test_data_backdoor, save_cuda = True)
        error_record.append(errors)
    else:
        error_record = []
        _, errors = MAE_error(backdoored_encoder, model, test_data_clean)
        error_record.append(errors)
        _, errors = MAE_error(backdoored_encoder, model, test_data_backdoor)
        error_record.append(errors)

    X = np.concatenate((np.array(error_record[0]), np.array(error_record[1])), axis=0)
    y = [0] * len(error_record[0]) + [1] * len(error_record[1])
    df = pd.DataFrame({'errors': X, 'bd_label': y})
    print('df shapes', df.shape)

    threshold1 = df['errors'].median()
    df['pred'] = df['errors'] > threshold1
    print("threshold:", threshold1, ".  sample detection accuracy", np.sum(df['pred'] == df['bd_label']) / len(df))
    threshold2 = df['errors'].mean()
    df['pred'] = df['errors'] > threshold2
    print("threshold:", threshold2, ".  sample detection accuracy", np.sum(df['pred'] == df['bd_label']) / len(df))
    threshold3 = df['errors'].quantile(.9)
    df['pred'] = df['errors'] > threshold3
    print("threshold:", threshold3, ".  sample detection accuracy", np.sum(df['pred'] == df['bd_label']) / len(df))
    threshold4 = df['errors'].quantile(1-args.poison_rate)
    df['pred'] = df['errors'] > threshold4
    print("threshold:", threshold4, ".  sample detection accuracy", np.sum(df['pred'] == df['bd_label']) / len(df))

    threshold = threshold1
    print('threshold used', threshold)
    nn_train_loader, _ = sift_dataset(backdoored_encoder, model, shadow_data, threshold)
    nn_test_loader, _ = sift_dataset(backdoored_encoder, model, test_data_clean, threshold)
    nn_backdoor_loader, num = sift_dataset(backdoored_encoder, model, test_data_backdoor, threshold)

    # main loop - after cleanse
    input_size = feature_bank_training.shape[1]
    criterion = nn.CrossEntropyLoss()
    net = NeuralNet(input_size, [512, 256], 10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)

    epochs = args.epochs
    ba_acc, asr_acc = [], []
    for epoch in range(epochs):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)

        acc1 = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        acc2 = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
        acc3 = (num[0] * acc2)/(num[0] + num[1]) # only clean idx data gets through the detector!
        ba_acc.append(acc1)
        asr_acc.append(acc3)

    result_record['ca_def'].append(ba_acc)
    result_record['asr_def'].append(asr_acc)

    print("BA best:", max(ba_acc), "ASR best:", max(asr_acc))
    with open('./plot_result/downstm_att{}_pr{}.pkl'.format(args.attack_type, args.poison_rate), 'wb') as f:  # open a text file
        pickle.dump(result_record, f)  # serialize the list