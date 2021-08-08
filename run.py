import argparse
import os, random
from sklearn.metrics.pairwise import cosine_similarity
import torch
import wandb
import yaml
from tqdm import tqdm
import utils
from model import SkipGram, NegativeSamplingLoss
from time import sleep
from torch.optim import Adam
from torch.utils.data import DataLoader
import eval.evaluation as evaluation

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
        '--VERSION', dest='VERSION',
        help='model version',
        type=int
    )
    parser.add_argument(
        '--CKPT_E', dest='CKPT_EPOCH',
        help='checkpoint epoch',
        type=int
    )
    parser.add_argument(
        '--NGRAMS', dest='NGRAMS',
        help='adding ngrams to tokens',
        action='store_true'
    )
    parser.add_argument(
        '--RESUME', dest='RESUME',
        help='resume training',
        action='store_true'
    )
    parser.add_argument(
        '--DATA', dest='DATA',
        help='location of dataset',
        type=str
    )
    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self, args, config):
        self.args = args
        self.cfgs = config
        self.loss = None
        self.batch_loss = None
        self.subsampling = True if self.args.SUBSAMPLING else False
        self.ngrams = True if self.args.NGRAMS else False

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if self.args.VERSION is None:
            self.model_ver = str(random.randint(0, 99999999))
        else:
            self.model_ver = str(self.args.VERSION)

        print("Model version:", self.model_ver)

        # Fix seed
        self.seed = int(self.model_ver)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)

    def train(self):
        dataset = utils.Dataset(self.args, self.cfgs)
        tokens = dataset.get_tokens('train')
        vocab = dataset.get_vocab_cls()
        vocab_size = len(vocab)
        ng_dist = utils.get_noise_dist(tokens)
        dataloader = utils.DataLoader(dataset, self.cfgs)
        data_size = len(tokens)
        print('Total data instances: ', data_size)
        model = SkipGram(self.cfgs, vocab_size, ng_dist).to(self.device)
        loss_func = NegativeSamplingLoss(model, self.cfgs).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.cfgs['LEARNING_RATE'])

        if self.args.RESUME:
            print('Resume training...')
            start_epoch = self.args.CKPT_EPOCH
            print('Loading Model ...')
            path = os.path.join(os.getcwd(), 'models',
                                self.model_ver,
                                'epoch' + str(start_epoch) + '.pkl')

            # Load state dict of the model and optimizer
            ckpt = torch.load(path, map_location=self.device)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
        else:

            start_epoch = 0
            os.mkdir(os.path.join(os.getcwd(), 'models', self.model_ver))

        model.train()
        print('Training started ...')
        print_every = 20
        for epoch in range(start_epoch, self.cfgs['EPOCHS']):
            loss_sum = 0
            with tqdm(dataloader.get_batches()) as tepoch:
                for step, (
                        input_words, target_words
                ) in enumerate(tepoch):
                    # for input_words, target_words in dataloader.get_batches():
                    tepoch.set_description("Epoch {}".format(str(epoch)))
                    # inputs, targets = torch.LongTensor(input_words).to(self.device),torch.LongTensor(target_words).to(self.device)

                    # Input contains list of different length, therefore cannot be converted into LongTensor at this point

                    targets = torch.LongTensor(target_words).to(self.device)
                    optimizer.zero_grad()
                    loss = loss_func(input_words, targets)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item()

                    tepoch.set_postfix(loss=loss.item())
                    sleep(0.1)

            utils.show_learning(model, vocab, self.device)
            self.loss = loss_sum/data_size
            self.batch_loss = loss_sum
            wandb.log({'batch_loss': self.batch_loss, 'loss': self.loss})
            if epoch % print_every == 0:
                self.eval(vocab, model.out_embeddings)
            epoch_finish = epoch + 1
            # Save checkpoint
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'embeds':model.in_embeddings,
                'vocab':vocab
            }

            torch.save(
                state,
                os.path.join(os.getcwd(), 'models',
                             self.model_ver,
                             'epoch' + str(epoch_finish) + '.pkl')
            )



    def eval(self, vocab_ins = None, embeds = None):
        if self.args.RUN_MODE == 'val':
            if self.args.CKPT_EPOCH is not None:
                path = os.path.join(os.getcwd(),'models',
                                    self.model_ver,
                                    'epoch' + str(self.args.CKPT_EPOCH) + '.pkl')
                # Load state dict of the model
                ckpt = torch.load(path, map_location=self.device)
                embeddings = ckpt['embeds']
                vocab = ckpt['vocab']
                print(vocab.get_vocab()[:10])
            else:
                print('CHECKPOINT not provided')
                exit(-1)
        else:
            vocab = vocab_ins
            embeddings = embeds
        evaluation.semantic_similarity_datasets(embeddings, vocab)



    def overfit(self):
        dataset = utils.Dataset(args, config)
        data = dataset.get_data(split=args.RUN_MODE)
        vocab = dataset.get_vocab(split=args.RUN_MODE)
        ng_dist = utils.get_noise_dist(data)
        dataloader = DataLoader(dataset, config['BATCH_SIZE'])

        data_size = len(vocab)
        model = SkipGram(self.cfgs, data_size, ng_dist).to(self.device)
        loss_func = NegativeSamplingLoss(model, self.cfgs).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.cfgs['LEARNING_RATE'])

        model.train()
        input_words, target_words = next(iter(dataloader))
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
    wandb.init(project='word_embeddings', entity='we')
    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)

    m_exec = MainExec(args, config)
    m_exec.run(args.RUN_MODE)
