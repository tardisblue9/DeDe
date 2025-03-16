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
import utils
from DRUPE.models.simclr_model import SimCLR
from DRUPE.datasets.cifar10_dataset import get_shadow_cifar10
from DECREE.imagenet import getBackdoorImageNet, get_processing
from DECREE.models import get_encoder_architecture_usage
from BadCLIP.pkgs.openai.clip import load as load_model

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_size = 10, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

def compute_error(model, dataset):
    model.eval()
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4) # sub to this if clip
    full_ce = nn.CrossEntropyLoss(reduction='none')

    res = []
    for i, batch in enumerate(tqdm(dataloader)):
        data, target = batch[0].cuda(), batch[1].cuda()
        with torch.no_grad():
            outputs = model(data)
            loss = full_ce(outputs, target)
            res.extend(loss.cpu().detach().numpy())

    return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train decoder detector on the given backdoored encoder')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train the decoder')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu the code runs on')

    parser.add_argument('--attack_type', type=str, default='badencoder')
    parser.add_argument('--poison_rate', default=0.1, type=float, help='training dataset poison rate, or shadow dataset')
    parser.add_argument('--test_poison_rate', default=0.5, type=float, help='test dataset poison rate, 0.01 ~ 0.5')

    args = parser.parse_args()


    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet18()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    full_ce = nn.CrossEntropyLoss(reduction='none')
    bce = torch.nn.MSELoss()

    ### load victim encoder
    if args.attack_type == 'badencoder':
        encoder_dir = './DRUPE/DRUPE_results/badencoder/pretrain_cifar10_sf0.2/downstream_cifar10_t12/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        o_model = vic_model.f
        o_model_input_size = 3072
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'drupe':
        encoder_dir = './DRUPE/DRUPE_results/drupe/pretrain_cifar10_sf0.2/downstream_cifar10_t12/' + 'epoch120.pth'
        checkpoint = torch.load(encoder_dir)
        vic_model = SimCLR().cuda()
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        o_model = vic_model.f
        o_model_input_size = 3072
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'ctrl':
        with open('./CTRL/args.pkl', 'rb') as handle:
            ctrl_args = pickle.load(handle)
        ctrl_args.data_path = './data/'
        ctrl_args.threat_model = 'our'
        print("load attack args:", ctrl_args)
        vic_model = set_model(ctrl_args).cuda()
        encoder_dir = './CTRL/Experiments/cifar10-simclr-resnet18-0.01-100.0-512-0.06-False-our-0/'  + 'last.pth.tar' #  + 'epoch_81.pth.tar'
        checkpoint = torch.load(encoder_dir, map_location='cpu')
        vic_model.load_state_dict(checkpoint['state_dict'], strict=False)
        o_model = vic_model.backbone
        o_model_input_size = 8192
        print("load backdoor model from", encoder_dir)
    elif args.attack_type == 'clip':
        with open('./DECREE/clip_text.pkl', 'rb') as handle:
            clip_args = pickle.load(handle)
        # args.pretrained_encoder = f'./output/CLIP_text/cifar10_backdoored_encoder/model_49clip_txt_atk_42.pth'
        clip_args.pretrained_encoder = f'./DECREE/output/CLIP_text/cifar10_backdoored_encoder/model_69clip_text_atk0.05_41.pth'
        vic_model = get_encoder_architecture_usage(clip_args).cuda()
        checkpoint = torch.load(clip_args.pretrained_encoder, map_location='cpu', weights_only=True)
        vic_model.visual.load_state_dict(checkpoint['state_dict'])
        o_model = vic_model.visual
        args.arch = 'CLIP' # assert
        args.image_size = 224 # assert
        o_model_input_size = 1024
        print("load backdoor model from", clip_args.pretrained_encoder)
    elif args.attack_type == 'badclip':
        vic_model, processor = load_model(name='RN50', pretrained=False)
        vic_model.cuda()
        state_dict = vic_model.state_dict()
        checkpoint = torch.load('./BadCLIP/logs/nodefence_ours_final/checkpoints/epoch_10.pt', map_location='cpu',
                                weights_only=False)
        state_dict_load = checkpoint["state_dict"]
        assert len(state_dict.keys()) == len(state_dict_load.keys())
        for i in range(len(state_dict.keys())):
            key1 = list(state_dict.keys())[i]
            key2 = list(state_dict_load.keys())[i]
            assert key1 in key2
            state_dict[key1] = state_dict_load[key2]
        vic_model.load_state_dict(state_dict)
        o_model = vic_model.visual
        args.arch = 'CLIP'  # assert
        args.image_size = 224  # assert
        o_model_input_size = 1024
        print("load backdoor model of badclip")
    else:
        print("invalid mode")
        1/0

    ### prepare train dataset for offset
    batch_size = 128
    nworkers = 8
    valid_size = 1000

    if args.attack_type == 'badencoder' or args.attack_type == 'drupe':
        aux_args = copy.deepcopy(args)
        aux_args.data_dir = './data/cifar10/'
        aux_args.trigger_file = './DRUPE/trigger/trigger_pt_white_21_10_ap_replace.npz'
        aux_args.reference_file = './DRUPE/reference/cifar10_l0.npz'  # depending on downstream tasks
        aux_args.reference_label = 0
        aux_args.shadow_fraction = args.poison_rate
        aux_args.dataset = 'cifar10'

        shadow_data = utils.CIFAR10_BACKDOOR(root='./data', train=True, trigger_file=aux_args.trigger_file,
                                             test_transform=utils.test_transform, poison_rate=args.poison_rate, lb_flag='')
        _, memory_data, test_data_clean, test_data_backdoor = get_shadow_cifar10(aux_args)

        train_dataloader = torch.utils.data.DataLoader(shadow_data, batch_size=batch_size, num_workers=nworkers, pin_memory=True, shuffle=True) # training poisoned dataset
        val_set, _ = torch.utils.data.random_split(memory_data, [0.1, 0.9]) # clean base dataset (refer to ASSET)
        poison_rate = args.test_poison_rate
        subset_test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor, [poison_rate, 1 - poison_rate])
        test_poi_set = ConcatDataset([test_data_clean, subset_test_data_backdoor]) # (the detection object)
    elif args.attack_type == 'ctrl':
        ctrl_args.poison_ratio = args.poison_rate
        train_loader, train_sampler, train_dataset, ft_loader, ft_sampler, test_loader, test_dataset, memory_loader, train_transform, ft_transform, test_transform = set_aug_diff(ctrl_args)
        poison_frequency_agent = PoisonFre(ctrl_args, ctrl_args.size, ctrl_args.channel, ctrl_args.window_size, ctrl_args.trigger_position, False, True)
        poison = PoisonAgent(ctrl_args, poison_frequency_agent, train_dataset, test_dataset, memory_loader, ctrl_args.magnitude)
        test_loader = poison.test_loader
        test_pos_loader = poison.test_pos_loader

        test_data_clean = test_loader.dataset
        test_data_backdoor = test_pos_loader.dataset
        memory_data = memory_loader.dataset
        shadow_data = train_dataset
        train_dataloader = torch.utils.data.DataLoader(shadow_data, batch_size=batch_size, num_workers=nworkers,
                                                       pin_memory=True, shuffle=True)
        val_set, _ = torch.utils.data.random_split(memory_data, [0.1, 0.9])  # clean base dataset (refer to ASSET)
        poison_rate = args.test_poison_rate
        subset_test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor, [poison_rate, 1 - poison_rate])
        test_poi_set = ConcatDataset([test_data_clean, subset_test_data_backdoor])  # (the detection object)
    elif args.attack_type == 'clip':
        # train_transform, _ = get_processing('imagenet', augment=True)
        # test_transform, _ = get_processing('imagenet', augment=False)
        # shadow_data = getBackdoorImageNet(
        #     trigger_file=clip_args.trigger_file,
        #     train_transform=test_transform,
        #     test_transform=test_transform,
        #     reference_word=clip_args.reference_word,
        #     poison_rate=args.poison_rate)
        batch_size = 4 # memory cuda
        valid_size = 100
        trigger_file = './DECREE/trigger/' + 'trigger_pt_white_185_24.npz'
        shadow_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=True, trigger_file=trigger_file,
                                             test_transform=utils.test_transform224, poison_rate=args.poison_rate,
                                             lb_flag='')
        memory_data = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,
                                               test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        memory_data2 = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,
                                                  test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        shadow_data, _ = torch.utils.data.random_split(shadow_data, [0.1, 0.9])
        memory_data, _ = torch.utils.data.random_split(memory_data, [0.1, 0.9])
        memory_data2, _ = torch.utils.data.random_split(memory_data2, [0.1, 0.9])
        train_dataloader = torch.utils.data.DataLoader(shadow_data, batch_size=batch_size, num_workers=nworkers,
                                                       pin_memory=True, shuffle=True)
        test_data_clean = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,
                                               test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_CLIP(root='./data', train=False, trigger_file=trigger_file,
                                                  test_transform=utils.test_transform, poison_rate=1, lb_flag='')

        test_data_clean, _ = torch.utils.data.random_split(test_data_clean, [0.1, 0.9])
        test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor, [0.1, 0.9])

        val_set, _ = torch.utils.data.random_split(memory_data, [0.1, 0.9])  # clean base dataset (refer to ASSET)
        val_set2, _ = torch.utils.data.random_split(memory_data2, [0.1, 0.9])
        test_poison_rate = args.test_poison_rate
        subset_test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor, [test_poison_rate, 1 - test_poison_rate])
        test_poi_set = ConcatDataset([test_data_clean, subset_test_data_backdoor])  # (the detection object)
        # print("dataset size:", len(shadow_data))
    elif args.attack_type == 'badclip':
        batch_size = 4  # memory cuda
        valid_size = 100
        shadow_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                                     test_transform=utils.test_transform224, poison_rate=args.poison_rate, lb_flag='backdoor')
        memory_data = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=True, trigger_file='',
                                               test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        memory_data2 = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                   test_transform=utils.test_transform224, poison_rate=0, lb_flag='')
        memory_data2, _ = torch.utils.data.random_split(memory_data2, [0.1, 0.9])
        test_data_clean = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                   test_transform=utils.test_transform, poison_rate=0, lb_flag='')
        test_data_backdoor = utils.CIFAR10_BACKDOOR_BadCLIP(root='./data', train=False, trigger_file='',
                                                      test_transform=utils.test_transform, poison_rate=1, lb_flag='backdoor')
        train_dataloader = torch.utils.data.DataLoader(shadow_data, batch_size=batch_size, num_workers=nworkers,
                                                       pin_memory=True, shuffle=True)
        test_data_clean, _ = torch.utils.data.random_split(test_data_clean, [0.1, 0.9])
        test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor, [0.1, 0.9])

        val_set, _ = torch.utils.data.random_split(memory_data, [0.1, 0.9])  # clean base dataset (refer to ASSET)
        val_set2, _ = torch.utils.data.random_split(memory_data2, [0.1, 0.9])
        test_poison_rate = args.test_poison_rate
        subset_test_data_backdoor, _ = torch.utils.data.random_split(test_data_backdoor,
                                                                     [test_poison_rate, 1 - test_poison_rate])
        test_poi_set = ConcatDataset([test_data_clean, subset_test_data_backdoor])  # (the detection object)
    else:
        print("invalid dataset")
        1/0
    print("size of val_set", len(val_set))
    print("size of test_poi_set", len(test_poi_set))


    # main loop
    roc_auc_scores = []
    results_record = {'roc_auc_score': [], 'accuracy': [], 'fpr_tpr_plot': [], 'fpr/tpr': [], 'clean_res':[], 'poi_res':[]}
    for epoch in tqdm(range(args.epochs)):
        o_model2 = copy.deepcopy(o_model)
        o_model2.train()

        model_hat = copy.deepcopy(o_model2)
        layer_cake = list(model_hat.children())
        # model_hat = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()) # comment this line when using clip
        model_hat = model_hat.to(device)
        model_hat = model_hat.train()
        model.train()

        for iters, batch in enumerate(train_dataloader):
            # pos_img, pos_lab = input_train.cuda(), target_train.cuda()
            pos_img = batch[0].cuda()
            pos_lab = batch[1].cuda()
            idxs = random.sample(range(valid_size), min(batch_size, valid_size))
            neg_img = torch.stack([val_set[i][0] for i in idxs]).to(device)
            neg_img2 = torch.stack([val_set2[i][0] for i in idxs]).to(device) # add this line when using clip
            neg_lab = torch.tensor([val_set[i][1] for i in idxs]).to(device)
            neg_outputs = model(neg_img)
            neg_loss = torch.mean(torch.var(neg_outputs, dim=1))
            optimizer.zero_grad()
            neg_loss.backward()
            optimizer.step()

            Vnet = MLP(input_size=o_model_input_size, hidden_size=512, num_layers=2).to(device)
            Vnet.train()
            optimizer_hat = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
            optimizer_hat2 = torch.optim.Adam(Vnet.parameters(), lr=0.0001)
            for _ in range(10):
                v_outputs = model_hat(pos_img)
                vneto = Vnet(v_outputs)
                v_label = torch.ones(v_outputs.shape[0]).to(device)
                rr_loss = bce(vneto.view(-1), v_label)
                Vnet.zero_grad()
                rr_loss.backward()
                optimizer_hat.step()

                vn_outputs = model_hat(neg_img2) # modify this line when using clip: neg_img2
                v_label2 = torch.zeros(vn_outputs.shape[0]).to(device)
                vneto2 = Vnet(vn_outputs)
                rr_loss2 = bce(vneto2.view(-1), v_label2)
                Vnet.zero_grad()
                rr_loss2.backward()
                optimizer_hat2.step()

            res = Vnet(v_outputs)
            pidx = torch.where(adjusted_outlyingness(res) > 2)[0]
            try:
                pos_outputs = model(pos_img[pidx])
            except:
                continue
            real_loss = -criterion(pos_outputs, pos_lab[pidx])
            optimizer2.zero_grad()
            real_loss.backward()
            optimizer2.step()
            print(neg_loss, real_loss)

        # testing
        loss = compute_error(model, test_poi_set)
        pred_label = loss
        true_label = [0 for i in range(len(test_data_clean))] + [1 for i in range(len(subset_test_data_backdoor))]
        assert len(loss) == len(true_label)
        clean_res, poi_res = loss[0:len(test_data_clean)], loss[len(test_data_clean):]

        fpr, tpr, thersholds = roc_curve(true_label, pred_label)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc_score(true_label, pred_label))
        print('roc_auc_score' , roc_auc_scores[-1])
        results_record['roc_auc_score'].append(roc_auc_scores[-1])

        df = pd.DataFrame({'errors': pred_label, 'bd_label': true_label})

        threshold = df['errors'].median()  # df['X'].mean()
        df['pred'] = df['errors'] > threshold
        acc1 = np.sum(df['pred'] == df['bd_label']) / len(df)  # sample detection accuracy

        threshold = df['errors'].mean()  # df['X'].mean()
        df['pred'] = df['errors'] > threshold
        acc2 = np.sum(df['pred'] == df['bd_label']) / len(df)  # sample detection accuracy

        threshold = df['errors'].quantile(.9)
        df['pred'] = df['errors'] > threshold
        acc3 = np.sum(df['pred'] == df['bd_label']) / len(df)  # sample detection accuracy
        results_record['accuracy'].append((acc1, acc2, acc3))

        y = df['bd_label']
        y_hat = df['pred']
        fpr_manual, tpr_manual, _ = roc_curve(y, y_hat, pos_label=1)
        results_record['fpr/tpr'].append((tpr_manual[1],fpr_manual[1]))

        results_record['clean_res'].append(clean_res)
        results_record['poi_res'].append(poi_res)
        # plt.figure(figsize=(3, 1.5), dpi=300)
        # plt.hist(np.array(clean_res), bins=200, label='Clean', color="#5da1f0")
        # plt.hist(np.array(poi_res), bins=200, label='Poison', color="#f7d145")
        # plt.ylabel("Number of samples")
        # plt.xticks([])
        # plt.ylim(0, 500)
        # plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
        # plt.legend(prop={'size': 6})
        # plt.savefig('./plot_result/{}_asset_dist.jpg'.format(args.attack_type), dpi=260)

        results_record['fpr_tpr_plot'].append((fpr, tpr))
        # plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve with tpr {}, fpr {}'.format(tpr_manual[1],fpr_manual[1]))
        # plt.legend(loc="lower right")
        # plt.savefig('./plot_result/{}_asset_roc.jpg'.format(args.attack_type), dpi=260)

        # print('roc_auc_score list', roc_auc_scores)
        with open('./plot_result/result_att{}_poi{}_tpoi{}.pkl'.format(args.attack_type,args.poison_rate,args.test_poison_rate), 'wb') as f:  # open a text file
            pickle.dump(results_record, f) # serialize the list
