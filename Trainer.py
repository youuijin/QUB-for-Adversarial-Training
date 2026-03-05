import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from utils.model import ResNet18

from attack.PGD_attack import PGDAttack
from attack.FGSM_attack import FGSM_Attack

from datetime import datetime

class Trainer:
    def __init__(self, args):
        # paths
        self.log_dir = f'{args.log_dir}/seed{args.seed}'
        self.save_dir = f'{args.save_dir}/seed{args.seed}'

        # normalize
        self.mean = (0, 0, 0)
        self.std = (1, 1, 1)
        
        # train environment
        self.device = f'cuda:{args.device}' if args.device>=0 else 'cpu'
        self.epoch = args.epoch
        self.valid_epoch = args.valid_epoch
        self.batch_size = args.batch_size
        
        # Model - ResNet18
        self.model = ResNet18(n_class=10)
        self.model = self.model.to(self.device)

        # Model - None/FGSM/PGD
        if args.train_attack == 'FGSM':
            self.train_attack = FGSM_Attack(self.model, eps=args.train_eps, mean=self.mean, std=self.std, device=self.device)
            self.attack_desc = f'FGSM(eps{args.train_eps})'
        elif args.train_attack == 'PGD_Linf':
            self.train_attack = PGDAttack(self.model, eps=args.train_eps, alpha=args.train_alpha, iter=args.attack_iter, 
                                      mean=self.mean, std=self.std, device=self.device)
            self.attack_desc = f'PGD_Linf(eps{args.train_eps}_alpha{args.train_alpha}_iter{args.attack_iter})'
        else:
            self.train_attack = None
            self.attack_desc = 'Clean'

        
        # Dataset - CIFAR10
        self.set_dataloader(args)

        # train loss
        self.loss = args.loss
        self.loss_desc = args.loss
        self.QUB_opt = args.QUB_opt
        if args.loss == 'QUB' and args.QUB_opt is not None:
            self.loss_desc = args.QUB_opt
        
        # QUB hyperparameter
        self.K = 0.5

        # Validation Attack - PGD10
        self.valid_attack = PGDAttack(self.model, eps=args.valid_eps, alpha=2., iter=10, mean=self.mean, std=self.std, device=self.device)

        # optimizer & lr scheduler (multistep)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        lr_steps = args.epoch * len(self.train_loader)
        decay_epochs = args.decay_epochs.split(',')
        milestones = [int(lr_steps*int(epoch)/args.epoch) for epoch in decay_epochs]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        # tracking best accuracy 
        self.best_acc, self.best_adv_acc, self.last_acc, self.last_adv_acc = 0, 0, 0, 0

        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'{self.attack_desc}_{self.loss_desc}_lr{args.lr}_{cur}'
        self.writer = SummaryWriter(f'{self.log_dir}/{self.log_name}')

    def set_dataloader(self, args):
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std)])
        transform_val = transforms.Compose([transforms.Resize(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])

        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_data = CIFAR10(root='./data', train=False, download=True, transform=transform_val)

        self.train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    def train(self):
        # Setting logger
        self.writer = SummaryWriter(f'{self.log_dir}/{self.log_name}')

        # train all steps
        for epoch in range(self.epoch):
            train_acc, train_loss = self.train_1_epoch(epoch) # implement in child class trainer

            # logging
            self.writer.add_scalar('train/acc', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)

            print(f'Epoch {epoch}: accuracy {round(train_acc, 2)}\tloss {round(train_loss, 4)}')

            if epoch%self.valid_epoch == 0:
                valid_acc, valid_adv_acc, valid_loss, valid_adv_loss = self.valid()
                # logging
                self.writer.add_scalar('valid/acc', valid_acc, epoch)
                self.writer.add_scalar('valid/adv_acc', valid_adv_acc, epoch)
                self.writer.add_scalar('valid/loss', valid_loss, epoch)
                self.writer.add_scalar('valid/adv_loss', valid_adv_loss, epoch)

        return self.best_acc, self.best_adv_acc, self.last_acc, self.last_adv_acc

    def train_1_epoch(self, epoch):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.loss == 'CE':
                adv_inputs = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                loss = F.cross_entropy(adv_outputs, targets)
            elif self.loss == 'QUB':
                outputs = self.model(inputs)
                softmax = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(targets, num_classes = softmax.shape[1])

                adv_inputs = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                adv_norm = torch.norm(adv_outputs-outputs, dim=1)

                loss = F.cross_entropy(outputs, targets, reduction='none')

                if self.QUB_opt == "QUBAT":
                    lamb = epoch/self.epoch
                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    adv_CE_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
                    loss = (1-lamb)*upper_loss + lamb*adv_CE_loss
                    loss = loss.mean()
                else:
                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    loss = upper_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_tot += loss.item()*targets.size(0)
            _, predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            self.scheduler.step()

            break

        train_acc = 100.*correct/total
        train_loss = loss_tot/total
        return train_acc, train_loss
    
    def valid(self):
        # valid 1 step
        self.model.eval()
        correct, adv_correct, loss_tot, adv_loss_tot, total = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # clean sample (SA)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                
                loss_tot += F.cross_entropy(outputs, targets, reduction='sum').item()

                # attacked sample (RA)
                adv_inputs = self.valid_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                _, adv_predicted = adv_outputs.max(1)
                adv_correct += adv_predicted.eq(targets).sum().item()

                adv_loss_tot += F.cross_entropy(adv_outputs, targets, reduction='sum').item()

                total += targets.size(0)

                break
            
        valid_acc = 100.*correct/total
        valid_adv_acc = 100.*adv_correct/total
        valid_loss = loss_tot/total
        valid_adv_loss = adv_loss_tot/total

        # Save checkpoint
        if valid_adv_acc > self.best_adv_acc:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_best.pt')
            self.best_acc = valid_acc
            self.best_adv_acc = valid_adv_acc

        self.last_acc = valid_acc
        self.last_adv_acc = valid_adv_acc
        torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_last.pt')

        return valid_acc, valid_adv_acc, valid_loss, valid_adv_loss