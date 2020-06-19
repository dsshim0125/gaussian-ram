import torch.utils.data
from torchvision import datasets, transforms
from model import GDRAM
from dataloader import MnistClutteredDataset
import time
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Inference')

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--fast', type=str2bool, default='False')
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

batch_size = 1

kwargs = {'num_workers': 64, 'pin_memory': True} if not args.device=='cpu' else {}

device = torch.device(args.device)

model_path = 'checkpoints/'+args.dataset+'_gdram.pth'

img_size = 128

torch.manual_seed(args.random_seed)

##################################################

if args.dataset == 'cifar10':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=False,\
                                                               transform=transform),batch_size=batch_size, shuffle=False, **kwargs)

elif args.dataset == 'cifar100':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor()])

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('data', train=False,\
                                                               transform=transform),batch_size=batch_size, shuffle=False, **kwargs)

elif args.dataset == 'mnist':

    transform = transforms.Compose([transforms.Resize(img_size),transforms.Grayscale(3), transforms.ToTensor()])
    test_set = MnistClutteredDataset('/media/dsshim/mnist_distortion', type='test',transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, **kwargs
    )


model = GDRAM(device=device, dataset=args.dataset, Fast=args.fast).to(device)
model.eval()

pytorch_total_params = sum(p.numel() for p in model.parameters())

print('Model parameters: %d'%pytorch_total_params)

model.load_state_dict(torch.load(model_path))
print('Model Loaded!')

total_correct = 0.0

def accuracy2(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


accuracy1 = 0
accuracy5 = 0

start_time = time.time()

for data, labels in test_loader:
    data = data.to(device)
    action_logits, location, _, _, weights = model(data)
    predictions = torch.argmax(action_logits, dim=1)
    labels = labels.to(device)
    total_correct += torch.sum((labels == predictions)).item()

    acc1 , acc5 = accuracy2(action_logits, labels, topk=(1,5))
    accuracy1 += acc1.detach().cpu().numpy()
    accuracy5 += acc5.detach().cpu().numpy()

acc1 = accuracy1/len(test_loader)
acc5 = accuracy5/len(test_loader)

print("Top1:%.2f Top5:%.2f fps:%.5f"%(acc1, acc5,(time.time() - start_time)/len(test_loader.dataset)))

