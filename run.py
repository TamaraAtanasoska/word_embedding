import argparse
import os
import torch
import yaml
from tqdm import tqdm
import utils
from model import SkipGram, NegativeSamplingLoss
from time import sleep
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter('/home/users/bverma/project/bhuvanesh/tmp/tensorboard_dirs/we')
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

    parser.add_argument(
        '--NGRAMS', dest='NGRAMS',
        help='adding ngrams to tokens',
        action='store_true'
    )
    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self, args, config):
        self.args = args
        self.cfgs = config
        self.subsampling = True if self.args.SUBSAMPLING else False
        self.ngrams = True if self.args.NGRAMS else False

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )


    def train(self):
        dataset = utils.Dataset(args)
        data = dataset.get_data(split = args.RUN_MODE)

        vocab = dataset.get_vocab_cls(split=args.RUN_MODE)
        ng_dist = utils.get_noise_dist(data)
        dataloader = utils.DataLoader(dataset, config['BATCH_SIZE'])
        
        data_size = len(vocab)
        print(data_size)
        model = SkipGram(self.cfgs, data_size, ng_dist).to(self.device)
        loss_func = NegativeSamplingLoss(model, self.cfgs).to(self.device)
        optimizer = Adam(model.parameters(), lr = self.cfgs['LEARNING_RATE'])
        running_loss = 0.0
        loss_sum = 0 
        model.train()
        print('Training started ...')
        for epoch in range(self.cfgs['EPOCHS']):
            with tqdm(dataloader.get_batches()) as tepoch:
                for step, (
                        input_words, target_words
                ) in enumerate(tepoch):
                    # for input_words, target_words in dataloader.get_batches():
                    tepoch.set_description("Epoch {}".format(str(epoch)))
                    #inputs, targets = torch.LongTensor(input_words).to(self.device),torch.LongTensor(target_words).to(self.device)

                    # Input contains list of different length, therefore cannot be converted into LongTensor at this point

                    targets = torch.LongTensor(target_words).to(self.device)
                    optimizer.zero_grad()
                    loss = loss_func(input_words, targets)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item()

                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)
                   # writer.add_scalar('training loss', loss.item(), epoch + step)
                    if step % 1000 == 0:
                        print("Epoch: {}/{}".format(epoch + 1, self.cfgs['EPOCHS']))
                        print("Loss: ", loss.item())  # avg batch loss at this point in training

            print('epoch {}, loss {}'.format(epoch, loss_sum/data_size))


    def eval(self):
        data = utils.Dataset(args)
        
        model = None
        loss_func = None
        dataloader = utils.Dataloader(dataset = data, split = args.RUN_MODE,
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
        running_loss = 0.0
        model.train()
        input_words, target_words = next(iter(dataloader.get_batches())) 
        inputs, targets = torch.LongTensor(input_words).to(self.device), \
                          torch.LongTensor(target_words).to(self.device)

        for epoch in range(self.cfgs['EPOCHS']):

            optimizer.zero_grad()
    
            loss = loss_func(inputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('epoch {}, loss {}'.format(epoch, round(loss.item(), 3)))
            #writer.add_scalar('training loss', round(loss.item(), 3) , epoch )

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
