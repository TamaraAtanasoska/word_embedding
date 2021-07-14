import argparse
import os
import torch
import yaml

import utils
from model import SkipGram, NegativeSamplingLoss

from torch.optim import Adam


def parse_args():

    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--RUN_MODE', dest='RUN_MODE',
        choices=['train', 'val'],
        help='{train, val}',
        type=str, required=True
    )

    parser.add_argument(
        '--CPU', dest='CPU',
        help='use CPU instead of GPU',
        action='store_true'
    )

    parser.add_argument(
        '--DEBUG', dest='DEBUG',
        help='enter debug mode',
        action='store_true'
    )

    parser.add_argument(
        '--SUBSAMPLING', dest='SUBSAMPLING',
        help='add subsampling to words',
        action='store_true'
    )
    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self, args, config):
        self.args = args
        self.cfgs = config
        self.subsampling = True if self.args.SUBSAMPLING else False

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )


    def train(self):
        dataset = utils.Dataset(args)
        data = dataset.get_data(split = args.RUN_MODE)
        ng_dist = utils.get_noise_dist(data)
        dataloader = utils.Dataloader(dataset = dataset, 
                                      split = args.RUN_MODE, 
                                      batch_size = self.cfgs['BATCH_SIZE'],
                                      subsampling = self.subsampling,
                                     )
        
        data_size = len(dataloader.tokens)
        model = SkipGram(self.cfgs, data_size, ng_dist).to(self.device)
        loss_func = NegativeSamplingLoss(model, self.cfgs).to(self.device)
        optimizer = Adam(model.parameters(), lr = self.cfgs['LEARNING_RATE'])
       
        loss_sum = 0 
        model.train()

        for epoch in range(self.cfgs['EPOCHS']):
            for input_words, target_words in dataloader.get_batches():
            
                inputs, targets = torch.LongTensor(input_words).to(self.device), \
                                  torch.LongTensor(target_words).to(self.device)

                optimizer.zero_grad() 
                loss = loss_func(inputs, targets)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
      
            print('epoch {}, loss {}'.format(epoch, loss_sum/data_size)) 

    def eval(self):
        data = utils.Dataset(args)
        
        model = None
        loss_func = None
        dataloader = utils.Dataloader(dataset = data, 
                                      batch_size = self.cfgs['BATCH_SIZE'],
                                     )


    def overfit(self):
        dataset = utils.Dataset(args)
        data = dataset.get_data(split = args.RUN_MODE)
        ng_dist = utils.get_noise_dist(data)
        dataloader = utils.Dataloader(dataset = dataset,
                                      split = args.RUN_MODE, 
                                      batch_size = self.cfgs['BATCH_SIZE'],
                                      subsampling = self.subsampling,
                                     )
        
        data_size = len(dataloader.tokens)
        model = SkipGram(self.cfgs, data_size, ng_dist).to(self.device)
        loss_func = NegativeSamplingLoss(model, self.cfgs).to(self.device)
        optimizer = Adam(model.parameters(), lr = self.cfgs['LEARNING_RATE'])
        
        model.train()

        input_words, target_words = next(iter(dataloader.get_batches())) 
        inputs, targets = torch.LongTensor(input_words).to(self.device), \
                          torch.LongTensor(target_words).to(self.device)

        for epoch in range(self.cfgs['EPOCHS']):
            optimizer.zero_grad()
    
            loss = loss_func(inputs, targets)
            loss.backward()
            optimizer.step()
        
            print('epoch {}, loss {}'.format(epoch, round(loss.item(), 3))) 
       

    def run(self, run_mode):
        if run_mode == 'train' and self.args.DEBUG:
            print('Overfitting a single batch...')
            self.overfit()
        elif run_mode == 'train':
            print('Starting training mode...')
            self.train()
        elif run_mode == 'val':
            print('Starting validation mode...')
            self.eval()
        else:
            exit(-1)


if __name__ == "__main__":
    args = parse_args()

    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)

    exec = MainExec(args, config)
    exec.run(args.RUN_MODE)
