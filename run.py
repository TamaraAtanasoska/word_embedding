import argparse
import os
import torch

import utils

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

    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self, args):
        self.args = args

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )


    def train(self):
        data = utils.Dataset(args)
        
        model = None
        loss_func = None
        optimizer = None
        dataloader = utils.Dataloader(dataset = data, 
                                      batch_size = 2,
                                      num_workers = 4,
                                     )


    def eval(self):
        data = utils.Dataset(args)
        
        model = None
        loss_func = None
        dataloader = utils.Dataloader(dataset = data, 
                                      batch_size = 2,
                                      num_workers = 4,
                                     )


    def overfit(self):
        data = utils.Dataset(args)
        
        model = None
        loss_func = None
        optimizer = None
        dataloader = utils.Dataloader(dataset=data, 
                                      batch_size=2,
                                      num_workers = 4,
                                     )
        batch = next(iter(dataloader))


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

    exec = MainExec(args)
    exec.run(args.RUN_MODE)
