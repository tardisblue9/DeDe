# This is a sample Python script.
import copy
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import argparse

import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torch.utils.data import Dataset, DataLoader
from einops import repeat, rearrange
from thop import profile, clever_format
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from DRUPE.models.simclr_model import SimCLR
from DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from CTRL.methods import set_model
from CTRL.loaders.diffaugment import set_aug_diff, PoisonAgent
from CTRL.utils.frequency import PoisonFre
from ASSET.models import ResNet18
from DECREE.imagenet import getBackdoorImageNet, get_processing
from DECREE.models import get_encoder_architecture_usage
from BadCLIP.pkgs.openai.clip import load as load_model

from decoder_model import DecoderModel
import utils


def train_decoder(backdoored_encoder, model, data_loader, optimizer, args):
    backdoored_encoder.eval()
    model.train()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for batch in train_bar:
        img = batch[0]
        img = img.cuda()
        feature_raw = backdoored_encoder(img)
        predicted_img, mask = model(img, feature_raw)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * args.batch_size
        total_num += args.batch_size
        train_bar.set_description('Decoder Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

def train_decoder_ood(backdoored_encoder, model, ood_data_loader, optimizer, args):
    backdoored_encoder.eval()
    model.train()
    if ood_data_loader == None:
        return 0

    total_loss, total_num, train_bar = 0.0, 0, tqdm(ood_data_loader)
    for batch in train_bar:
        img = batch[0]
        # z_img = torch.ones(img.shape).cuda() # torch.rand(img.shape).cuda() # mod for ctrl. Comment this line for other attacks
        img = img.cuda()
        feature_raw = backdoored_encoder(img)
        predicted_img, mask = model(img, feature_raw)
        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * args.batch_size
        total_num += args.batch_size
        train_bar.set_description('Decoder Train Epoch: [{}/{}] ood Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

def ood_dataset_seperate(train_dataloader, args):
    if args.attack_type == 'ctrl':
        backdoored_encoder.eval()
        with torch.no_grad():
            feature_bank = []
            target_bank = []

            test_bar = tqdm(train_dataloader)
            for data, target, _ in test_bar:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = backdoored_encoder(data)

                target_bank.append(target)
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
            target_bank = torch.cat(target_bank, dim=0).contiguous()

        feature_bank = feature_bank.detach().cpu().numpy()
        target_bank = target_bank.detach().cpu().numpy()

        pca = PCA(n_components=2)
        result = pca.fit_transform(feature_bank)
        print(result.shape)
        label = target_bank[:result.shape[0]]
        df = pd.DataFrame(result)
        df['label'] = label
        sns.scatterplot(data=df, x=0, y=1, hue="label", marker="+")
        plt.savefig('./plot_result/ctrl_train_pca.jpg', dpi=260)
        plt.show()

        feature_norm = np.linalg.norm(feature_bank, ord=2, axis=1)
        df['norm'] = feature_norm
        threshold = df['norm'].quantile(1-args.poison_rate)  # .mean()+2
        print('threshold', threshold , "by quantile", 1-args.poison_rate)
        df_sub = df[df['norm'] > threshold]

        sns.scatterplot(data=df_sub, x=0, y=1, hue="label", marker="+")
        plt.savefig('./plot_result/ctrl_train_pca_rmood.jpg', dpi=260)
        plt.show()

        ood_dataset = torch.utils.data.Subset(shadow_data, df_sub.index)
        id_dataset = torch.utils.data.Subset(shadow_data, df[df['norm'] < threshold].index)
        print("after filter, we have {} id data and {} ood data".format(len(id_dataset), len(ood_dataset)))

        train_dataloader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                      pin_memory=True, drop_last=True)
        ood_train_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                      pin_memory=True, drop_last=True)
        return train_dataloader, ood_train_dataloader
    else:
        return train_dataloader, None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--seed', default=21, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')

    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--encoder_dir', default='', type=str, help='path to the backdoored encoder')
    parser.add_argument('--poison_rate', default=0.1, type=float, help='learning rate in SGD')

    parser.add_argument('--mask_ratio', type=float, default=0.9)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--encoder_layer', type=int, default=12)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--traindata_type', type=str, default='id') # id: in distribution but poison as meta-data; ood: additional by choice
    parser.add_argument('--save_tag', type=str, default='')

    args = parser.parse_args()

    args.results_dir = "./DEDE_results/" + args.attack_type + '_' + args.save_tag  + "/"
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("exp args:", args)
    with open('{}/args.pkl'.format(args.results_dir), 'wb') as f:  # open a text file
        pickle.dump(args, f) # serialize the list

    # load encoder
    if args.attack_type == 'badencoder':
        args.encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t12/'
        encoder_dir = args.encoder_dir + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.f
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        args.encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t12/'
        encoder_dir = args.encoder_dir + 'epoch120.pth'
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
        vic_model = set_model(ctrl_args).cuda()
        ctrl_args.encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-0/' + 'last.pth.tar'
        checkpoint = torch.load(ctrl_args.encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        backdoored_encoder = vic_model.backbone
        print("load backdoor model from", ctrl_args.encoder_dir)
    elif args.attack_type == 'asset': # their baseline attack
        vic_model = ResNet18().cuda()
        vic_model.load_state_dict(torch.load('./ASSET/cifar10_backdoor_0.1_resnet18_tar2.pth', map_location='cpu'))
        layer_cake = list(vic_model.children())
        backdoored_encoder = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
        print("load backdoor model from", './ASSET/cifar10_backdoor_0.1_resnet18_tar2.pth')
    elif args.attack_type == 'clip':
        with open('./DECREE/clip_text.pkl', 'rb') as handle:
            clip_args = pickle.load(handle)
        clip_args.pretrained_encoder = f'./DECREE/output/CLIP_text/cifar10_backdoored_encoder/model_69clip_text_atk0.05_41.pth'

        vic_model = get_encoder_architecture_usage(clip_args).cuda()
        checkpoint = torch.load(clip_args.pretrained_encoder, map_location='cpu', weights_only=True)
        vic_model.visual.load_state_dict(checkpoint['state_dict'])
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP' # assert for decoder model
        args.image_size = 224 # assert for decoder model
        print("load backdoor model from", clip_args.pretrained_encoder)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        encoder_dir = './BadCLIP/logs/nodefence_ours_final/checkpoints/epoch_10.pt'
        checkpoint = torch.load(encoder_dir, map_location='cpu',
                                weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())): # match dict
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        backdoored_encoder = vic_model.visual
        args.arch = 'CLIP'  # assert for decoder model
        args.image_size = 224  # assert for decoder model
        print("load backdoor model from", encoder_dir)
    else:
        print("invalid mode")
        1/0

    # load corresponding datasets, if eligible
    if args.attack_type == 'badencoder' or args.attack_type == 'drupe' or args.attack_type == 'asset':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz' # depending on downstream tasks
        aux_args.shadow_fraction = args.poison_rate
        aux_args.reference_label = 0

        shadow_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='')
        _, memory_data, test_data_clean, test_data_backdoor = get_shadow_cifar10(aux_args)
        print("dataset size:", len(shadow_data))
    elif args.attack_type == 'ctrl':
        ctrl_args.poison_ratio = args.poison_rate
        train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(
            ctrl_args)
        poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size,
                                           ctrl_args.trigger_position, False, True)
        poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader,
                             ctrl_args.magnitude)
        train_pos_loader = poison.train_pos_loader
        test_loader = poison.test_loader
        test_pos_loader = poison.test_pos_loader

        train_data_poi_ut = train_pos_loader.dataset
        test_data_clean_ut = test_loader.dataset
        test_data_backdoor_ut = test_pos_loader.dataset
        train_data_poi = utils.DummyDataset(train_data_poi_ut, transform=utils.test_transform)
        test_data_clean = utils.DummyDataset(test_data_clean_ut, transform=utils.test_transform)
        test_data_backdoor = utils.DummyDataset(test_data_backdoor_ut, transform=utils.test_transform)

        memory_data = memory_loader.dataset
        shadow_data = train_data_poi
        print("dataset size:", len(shadow_data))
    elif args.attack_type == 'clip':
        train_transform, _ = get_processing('imagenet', augment=True)
        test_transform, _ = get_processing('imagenet', augment=False)
        shadow_data = getBackdoorImageNet(
            trigger_file=clip_args.trigger_file,
            train_transform=test_transform,
            test_transform=test_transform,
            reference_word=clip_args.reference_word,
            poison_rate=args.poison_rate)
        # shadow_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
        #                                      test_transform=utils.test_transform224, poison_rate=args.poison_rate,
        #                                      lb_flag='')
        trigger_file = './DECREE/trigger/' + 'trigger_pt_white_185_24.npz'
        memory_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,
                                               test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        print("dataset size:", len(shadow_data))
    elif args.attack_type == 'badclip':
        train_transform, _ = get_processing('imagenet', augment=True)
        test_transform, _ = get_processing('imagenet', augment=False)
        shadow_data = getBackdoorImageNet(
            trigger_file=clip_args.trigger_file,
            train_transform=test_transform,
            test_transform=test_transform,
            reference_word=clip_args.reference_word,
            poison_rate=args.poison_rate)
        trigger_file = ''
        # shadow_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file=trigger_file,
        #                                      test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='')
        memory_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file=trigger_file,
                                                  test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        print("dataset size:", len(shadow_data))
    else:
        print("invalid dataset")
        1/0

    # prepare training data
    if args.traindata_type == 'id':
        print("for in dist dataset, use shadow dataset")
        train_dataloader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    elif args.traindata_type == 'ood':
        print("for ood dataset, use unlabeled STL")
        train_data = STL10(root='./data', split='unlabeled', transform=utils.test_transform)  # resize 32 for cifar10 ; 224 for CLIP
        if args.attack_type == 'clip' or 'badclip': # 224 image size
            train_data = STL10(root='./data', split='unlabeled', transform=utils.test_transform224)  # resize 32 for cifar10 ; 224 for CLIP
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    print("training data size:", len(train_dataloader)*args.batch_size)
    train_dataloader, ood_train_dataloader = ood_dataset_seperate(train_dataloader, args) # invoked for ctrl

    # prepare decoder model
    model = DecoderModel(image_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio, encoder_layer=args.encoder_layer, decoder_layer=args.decoder_layer, arch=args.arch).cuda()

    # training
    results_record = {'train_loss': [], 'eval_img': [], 'train_loss2': []}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), torch.randn(512).cuda()))
    # flops, params = clever_format([flops, params])
    # print('# MAE Model Params: {} FLOPs: {}'.format(params, flops))

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_decoder(backdoored_encoder, model, train_dataloader, optimizer, args)
        train_loss2 = train_decoder_ood(backdoored_encoder, model, ood_train_dataloader, optimizer, args) # test
        img , _ = utils.MAE_test(backdoored_encoder, model, memory_data)  # genarate mae test images, for monitoring
        results_record['train_loss'].append(train_loss)
        results_record['train_loss2'].append(train_loss2)
        results_record['eval_img'].append(img)

        if len(results_record['eval_img']) > 10:
            results_record['eval_img'] = results_record['eval_img'][-10:]
        with open(args.results_dir + '/results_record.pkl', 'wb') as file:
            pickle.dump(results_record, file)

        if epoch % 10 == 0:
            torch.save(
                {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                args.results_dir + '/epoch' + str(epoch) + '.pth')
    torch.save(
        {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
        args.results_dir + '/final.pth')

