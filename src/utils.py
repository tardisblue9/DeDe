import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR10, GTSRB, SVHN
from PIL import Image
import random
from einops import repeat, rearrange
from BadCLIP.backdoor.utils import apply_trigger

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# dummy_transform = transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.ToTensor()])

test_transform224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def MAE_test(backdoored_encoder, mae_model, val_dataset, num2save = 16):
    ''' visualize the first 16 predicted images on val dataset'''
    backdoored_encoder.eval()
    mae_model.eval()
    with torch.no_grad():
        val_img = torch.stack([val_dataset[i][0] for i in range(num2save)])
        val_img = val_img.cuda()
        feature_raw = backdoored_encoder(val_img)
        predicted_val_img, mask = mae_model(val_img, feature_raw)
        predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)

        errors = []
        for i in range(val_img.shape[0]):
            error = torch.sum((val_img[i] - predicted_val_img[i]) ** 2)
            errors.append(error.detach().cpu().numpy())
    return img, errors


def MAE_error(backdoored_encoder, mae_model, val_dataset, save_cuda = False):
    backdoored_encoder.eval()
    mae_model.eval()
    num = len(val_dataset)
    if save_cuda is False:
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16,
                                    pin_memory=True, drop_last=False)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16,
                                    pin_memory=True, drop_last=False)
    errors = []
    imgs = []
    with torch.no_grad():
        for batch in val_dataloader:
            val_img = batch[0]
            val_img = val_img.cuda()
            feature_raw = backdoored_encoder(val_img)
            predicted_val_img, mask = mae_model(val_img, feature_raw)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)

            if save_cuda is False:
                imgs.append(img)
            else:
                pass

            for i in range(val_img.shape[0]):
                error = torch.sum((val_img[i] - predicted_val_img[i]) ** 2)
                errors.append(error.item())

    return imgs, errors



def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    # device = torch.device(f'cuda:{args.gpu}')
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion, keyword='Accuracy'):
    # device = torch.device(f'cuda:{args.gpu}')
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            # print('output:', output)
            # print('target:', target)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            # if 'ASR' in keyword:
            #     print(f'output:{np.asarray(pred.flatten().detach().cpu())}')
            #     print(f'target:{np.asarray(target.flatten().detach().cpu())}\n')
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc


def predict_feature(net, data_loader):
    net.eval()
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for batch in tqdm(data_loader, desc='Feature extracting'):
            data, target = batch[0], batch[1]
            feature = net(data.cuda())
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()



class CIFAR10_BACKDOOR(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = self.test_transform(Image.fromarray(img))

        if self.test_transform is not None:
            tg_mask = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_mask)).convert('RGB'))
            tg_patch = self.test_transform(
                Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB'))
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            if self.flag == 'backdoor':
                target = self.target_label

        return img, target

    def __len__(self):
        return len(self.source_dataset)

class CIFAR10_BACKDOOR_CLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t']
        self.trigger_mask = self.trigger_input_array['tm']
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = self.test_transform(Image.fromarray(img))

        tg_mask = np.uint8(self.trigger_mask).transpose(2, 0, 1)
        tg_patch = np.uint8(self.trigger_patch).transpose(2, 0, 1)
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            if self.flag == 'backdoor':
                target = self.target_label
        return img, target

    def __len__(self):
        return len(self.source_dataset)


class CIFAR10_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        self.source_dataset = CIFAR10(root=root, train=train, transform=None, download=True)
        #         self.trigger_input_array = np.load(trigger_file)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset.data[index], self.source_dataset.targets[index]
        img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)


class GTSRB_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        if train == True:
            self.source_dataset = GTSRB(root=root, split='train', transform=None, download=True)
        else:
            self.source_dataset = GTSRB(root=root, split='test', transform=None, download=True)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset[index][0], self.source_dataset[index][1]
        #         img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)

class SVHN_BACKDOOR_BadCLIP(Dataset):

    def __init__(self, root, train, trigger_file, test_transform, poison_rate, lb_flag):
        if train == True:
            self.source_dataset = SVHN(root=root, split='train', transform=None, download=True)
        else:
            self.source_dataset = SVHN(root=root, split='test', transform=None, download=True)
        self.test_transform = test_transform
        self.poison_rate = poison_rate
        self.poison_list = random.sample(range(len(self.source_dataset)),
                                         int(len(self.source_dataset) * poison_rate))
        self.flag = lb_flag
        if lb_flag == 'backdoor':
            self.target_label = 0  # 9

    def __getitem__(self, index):
        img, target = self.source_dataset[index][0], self.source_dataset[index][1]
        #         img = Image.fromarray(img)

        if index in self.poison_list:
            img = apply_trigger(img)
            if self.flag == 'backdoor':
                target = self.target_label
        else:
            pass

        img = self.test_transform(img)

        return img, target

    def __len__(self):
        return len(self.source_dataset)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DummyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y, z = self.data[index]
        x = (255 * x.permute(1, 2, 0).numpy()).astype(np.uint8)
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, z