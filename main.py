import utils
from argparse import Namespace

args = Namespace(CKPT_EPOCH=None, CPU=False, DEBUG=False, RESUME=False, RUN_MODE='val', VERSION=None)
dataset = utils.Dataset(args)
data = dataset.get_data(split='val')
noise_dist = utils.get_noise_dist(data)
dataloader = utils.Dataloader(dataset=dataset, split='val', batch_size=5)

print(next(dataloader.get_batches()))
print(noise_dist.shape)
# Better not use loop for now it will print a lot of data
'''
for input_words, target_words in dataloader.get_batches():
    print(input_words,target_words)
'''
