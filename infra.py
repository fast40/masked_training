import pathlib
import argparse
import logging

parser = argparse.ArgumentParser(__file__)
parser.add_argument('-d', '--data-dir', required=True, help='The parent directory of RUN_DIR.', type=pathlib.Path)
parser.add_argument('-r', '--run-dir', required=True, help='The directory to save data like model checkpoints, loss, and images to.', type=pathlib.Path)
parser.add_argument('-l', '--logfile', help='The file to log to.')
args = parser.parse_args()

# ----- logger -----

logging.basicConfig(filename=args.logfile, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(message)s')
logger = logging.getLogger(__name__)

logging.info(f'Program started.')

# ----- data dir -----

DATA_DIR = args.data_dir
RUN_DIR = DATA_DIR / args.run_dir

try:
    RUN_DIR.mkdir(parents=True)
except FileExistsError:
    logger.critical(f'{args.run_dir} already exists in {args.data_dir}. Please specify an empty location.')
    quit()

# ----- device -----

import torch

if torch.cuda.is_available():
    logger.info(f'CUDA is available')
    DEVICE = torch.device('cuda')
else:
    logger.info(f'CUDA is not available; falling back to CPU.')
    DEVICE = torch.device('cpu')

logger.info(f'Set DEVICE={DEVICE}')

# ----- data collection -----

import functools
import data_infra

DB_FILE = DATA_DIR / 'data.db'

logger.info(f'Initializing tables in {DB_FILE}')
data_infra.init_tables(DB_FILE)

logger.info(f'run_name={str(RUN_DIR)}')

add_metric = functools.partial(data_infra.add_metric, run_name=str(RUN_DIR), db_file=DB_FILE)
add_file = functools.partial(data_infra.add_file, run_name=str(RUN_DIR), db_file=DB_FILE)

