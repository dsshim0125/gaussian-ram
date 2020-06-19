import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from model import DRAM
from utils import draw_locations
from dataloader import MnistClutteredDataset

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Gaussian-RAM')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--lr', type=float, default='1e-3')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--resume', type=str2bool, default='False')
args = parser.parse_args()

assert (args.dataset=='mnist' or args.dataset=='cifar10') or args.dataset=='cifar100', 'please use dataset in mnist, cifar10 or cifar100'
torch.manual_seed(args.random_seed)

kwargs = {'num_workers': 32, 'pin_memory': True} if not args.device=='cpu' else {}

device = torch.device(args.device)


img_size = 128



##################################################

if args.dataset == 'cifar10':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])
    # training set : validation set : test set = 50000 : 10000 : 10000

    train_set = datasets.CIFAR10('data',train=True, download=True, transform=transform)
    indices = list(range(len(train_set)))
    valid_size = 10000
    train_size = len(train_set) - valid_size

    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False,\
                                                               transform=transform),batch_size=args.batch_size, shuffle=False, **kwargs)
if args.dataset == 'cifar100':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])
    # training set : validation set : test set = 50000 : 10000 : 10000

    train_set = datasets.CIFAR100('data',train=True, download=True, transform=transform)
    indices = list(range(len(train_set)))

    valid_size = 10000
    train_size = len(train_set) - valid_size

    train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=False,\
                                                               transform=transform),batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'mnist':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.Grayscale(3), transforms.ToTensor()])

    train_set = MnistClutteredDataset(type='train', transform=transform)
    valid_set = MnistClutteredDataset(type='val', transform= transform)
    test_set = MnistClutteredDataset(type='test',transform=transform)

    train_size = len(train_set)
    valid_size = len(valid_set)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, **kwargs
    )


model = GDRAM(device=device, dataset=args.dataset, Fast = False).to(device)

if args.resume:
    model.load_state_dict(torch.load(args.checkpoint))

pytorch_total_params = sum(p.numel() for p in model.parameters())

print('Model parameters: %d'%pytorch_total_params)


lr_decay_rate = args.lr / args.epochs
optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, patience=5)

predtion_loss_fn = nn.CrossEntropyLoss()

def loss_function(labels, action_logits, location_log_probs, baselines):

    pred_loss = predtion_loss_fn(action_logits, labels.squeeze())
    predictions = torch.argmax(action_logits, dim=1, keepdim=True)
    num_repeats = baselines.size(-1)
    rewards = (labels == predictions.detach()).float().repeat(1, num_repeats)


    baseline_loss = F.mse_loss(rewards, baselines)
    b_rewards = rewards - baselines.detach()
    reinforce_loss = torch.mean(
        torch.sum(-location_log_probs * b_rewards, dim=1))

    return pred_loss + baseline_loss + reinforce_loss


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        action_logits, loc, location_log_probs, baselines, _ = model(data)

        labels = labels.unsqueeze(dim=1).to(device)

        loss = loss_function(labels, action_logits, location_log_probs, baselines)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_size,
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / train_size))



def test(epoch, data_source, size):
    model.eval()
    total_correct = 0.0
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_source):
            data = data.to(device)
            action_logits, _,  _, _, _= model(data)
            predictions = torch.argmax(action_logits, dim=1)
            labels = labels.to(device)
            total_correct += torch.sum((labels == predictions)).item()
    accuracy = total_correct / size

    image = data[0:1]
    _, locations, _, _, weights = model(image)
    draw_locations(image.cpu().numpy()[0], locations.detach().cpu().numpy()[0], weights=weights, epoch=epoch)
    return accuracy


best_valid_accuracy, test_accuracy = 0, 0

for epoch in range(1, args.epochs + 1):
    accuracy = test(epoch, valid_loader, valid_size)
    scheduler.step(accuracy)
    print('====> Validation set accuracy: {:.2%}'.format(accuracy))
    if accuracy > best_valid_accuracy:
        best_valid_accuracy = accuracy
        test_accuracy = test(epoch, test_loader, len(test_loader.dataset))

        #torch.save(model.state_dict(), 'checkpoints/' + args.dataset + '_rnn_adaptive_12_test.pth')

        print('====> Test set accuracy: {:.2%}'.format(test_accuracy))
    train(epoch)

print('====> Test set accuracy: {:.2%}'.format(test_accuracy))
