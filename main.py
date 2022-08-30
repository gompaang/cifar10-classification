import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

from models.alexnet import *
from models.vggnet import *
from models.resnet import *

import wandb
wandb.init(project="cifar10-batch", entity="heystranger")
wandb.config = {
  "learning_rate": 0.01,
  "epochs": 50,
  "batch_size": 64
}

def main():
    parser = argparse.ArgumentParser(description="CIFAR10 image-classification")
    parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--epoch', default=50, type=int, help='')
    parser.add_argument('--train_batchsize', default=64, type=int, help='')
    parser.add_argument('--test_batchsize', default=64, type=int, help='')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='')
    args = parser.parse_args()

    start = Start(args)
    start.run()


class Start(object):
    def __init__(self, config):
        self.model = None    #설정해주기
        self.device = None
        self.cuda = config.cuda
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batchsize = config.train_batchsize
        self.test_batchsize = config.test_batchsize
        self.criterion = None
        self.optimizer = None
        self.trainloader = None
        self.testloader = None

    #데이터 불러오기
    def data_load(self):
        print('preparing data...')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # RGB의 mean과 RGB의 std
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(trainset, batch_size=self.train_batchsize, shuffle=True, num_workers=2)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(testset, batch_size=self.test_batchsize, shuffle=True, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #모델 설정
    def model_load(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = VGG16().to(self.device)      # 모델 설정
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.0005, momentum=0.9)

    #training
    def train(self):
        self.model.train()

        train_loss = 0
        train_correct = 0
        total = 0
        train_acc = 0
        pbar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predict = outputs.max(1, keepdim=True)[1]
            total += targets.size(0)

            train_correct += predict.eq(targets.view_as(predict)).sum().item()
            train_acc = train_correct / total
            pbar.set_description(f'[Train] {train_acc:.3f}')

        return train_loss, train_acc

    #testing
    def test(self):
        print('testing..')
        self.model.eval()

        test_loss = 0
        test_correct = 0
        total = 0
        test_acc = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                predict = outputs.max(1, keepdim=True)[1]
                total += targets.size(0)

                test_correct += predict.eq(targets.view_as(predict)).sum().item()
                test_acc = test_correct / total

        return test_loss, test_acc

    def save(self):
        model_savepath = 'pthbatch/2/model_vggnet16.pth'      # 돌릴때마다 이름 바꿔주기
        torch.save(self.model, model_savepath)
        print(f"checkpoint saved to {model_savepath}")

    def run(self):
        self.data_load()
        self.model_load()
        accuracy = 0
        start_time = time.time()

        for epoch in range(1, self.epochs+1):
            print(f'\n---------- epoch: {epoch}')

            train_result = self.train()
            total_train_time = (time.time() - start_time) / 60
            test_result = self.test()

            #wandb.log({"train_loss": train_result[0], "train_accuracy": train_result[1]})
            #wandb.log({"valid_loss": test_result[0], "valid_accuracy": test_result[1]})
            wandb.log({"loss": {"train_loss": train_result[0], "valid_loss": test_result[0]},
                       "accuracy": {"train_accuracy": train_result[1], "valid_accuracy": test_result[1]}})

            wandb.watch(self.model)

            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print(f'\n===> best accuracy :{accuracy}%')
                print(f"===> total training time: {total_train_time:.2f} min\n")
                self.save()

        #print(summary(self.model, (3, 32, 32)))


if __name__ == '__main__':
    main()