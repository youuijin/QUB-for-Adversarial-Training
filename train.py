import argparse, csv
from utils.utils import set_seed
from Trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()

    # environment options
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    # path options
    parser.add_argument('--log_dir', type=str, default='./results/logs', help='path to log by Summarywriter')
    parser.add_argument('--save_dir', type=str, default='./results/saved_models', help='path to save checkpoint models, Error when the folder not defined')
    parser.add_argument('--csv_name', type=str, default='./results/csvs')

    # model options
    parser.add_argument('--model', type=str, default='resnet18')

    # dataset options
    parser.add_argument('--dataset', default='cifar10')

    # train options
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--valid_epoch', type=int, default=5, help='validation interval')
    parser.add_argument('--decay_epochs', type=str, default='70,85', help='if you choose multistep scheduler, put epochs to decay split by (,)')
    parser.add_argument('--base_lr', type=float, default=0.0, help='learning rate')

    # Adversarial Training options
    parser.add_argument('--train_attack', type=str, default='None', choices=['None', 'FGSM', 'PGD_Linf'])
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'QUB'])
    parser.add_argument('--QUB_opt', type=str, default=None, choices=['QUBAT'])

    ## attack options
    parser.add_argument('--train_eps', type=float, default=8.0)
    parser.add_argument('--train_alpha', type=float, default=2.0)
    parser.add_argument('--attack_iter', type=int, default=10)

    # 기본 파싱
    args, _ = parser.parse_known_args()

    # validation options
    parser.add_argument('--valid_eps', type=float, default=8.)

    args = parser.parse_args()
    
    return args

# OOP structure
if __name__ == '__main__':
    args = get_args()
    set_seed(seed=args.seed)
    trainer = Trainer(args)

    # train
    print("Start Training", trainer.log_name)
    best_acc, best_adv_acc, last_acc, last_adv_acc = trainer.train()

    print(f'Finish Training\nlog name: {trainer.log_name}\nbest acc:{best_acc}%  best adv acc:{best_adv_acc}% ')
    file_name = f'{args.csv_name}.csv'
    with open(file_name, 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([trainer.log_name, best_acc, best_adv_acc, last_acc, last_adv_acc])