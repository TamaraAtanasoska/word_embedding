from typing import Any

import wandb
import sys
import run
import yaml
from argparse import Namespace

with open('./config.yml', 'r') as f:
    config = yaml.safe_load(f)
def maybe_convert_to_numeric(v: Any):
    try:
        return int(v)  # this will execute if string v is integer
    except ValueError:
        pass
    try:
        return float(v)  # this will execute if string v is float
    except ValueError:
        pass

    return v

argv = sys.argv[1:]

for arg in argv:
    arg = arg[2:]
    key, value = arg.split('=')
    value = maybe_convert_to_numeric(value)
    config[key] = value

args = Namespace(CKPT_EPOCH=None, CPU=False, DEBUG=False, RESUME=False, RUN_MODE='train', VERSION=None, SUBSAMPLING=True, NGRAMS=True, DATA='data/')
exec = run.MainExec(args, config)
exec.run(args.RUN_MODE)
#wandb.log({'batch_loss': exec.batch_loss, 'loss': exec.loss})