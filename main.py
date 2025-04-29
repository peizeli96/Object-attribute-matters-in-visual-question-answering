import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VQAFeatureDataset, Dictionary
from model import Model
import utils
import opt as opts
from train import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        random.seed(seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    opt.device = device


    batch_size = opt.batch_size

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    

    train_dset = VQAFeatureDataset(opt,'train', dictionary) 
    test_dset = VQAFeatureDataset(opt,'test', dictionary)  

    model = Model(test_dset,opt)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optim = None
    epoch = 0
    if opt.input is not None:
        print('loading %s' % opt.input)
        model.load_state_dict(torch.load(opt.input))
        model.to(device)


    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate,
                                  pin_memory=True)
    eval_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate,
                                 pin_memory=False)        

    
    train(opt, model, train_loader, eval_loader, opt.epochs, opt.output, epoch)
