import utils
from argparse import Namespace
args = Namespace(CKPT_EPOCH=None, CPU=False, DEBUG=False, RESUME=False, RUN_MODE='val', VERSION=None)
data = utils.Dataset(args)
dataloader = utils.Dataloader(dataset=data, batch_size=2)
print(next(dataloader.get_batches()))

## Better not use loop for now it will print alot of data
for input_words, target_words in dataloader.get_batches():
    print(input_words,target_words)