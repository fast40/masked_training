'''
ok so I just ran into the need to graph a number in matplotlib.

The workflow should literally be to pass the number to a function call. (as well as the label of the graph that's being used)
auto increment x axis.

I want this shit saved to disk.
'''



import pathlib
import argparse
import logging

parser = argparse.ArgumentParser(__file__)
parser.add_argument('-d', '--data-dir', required=True, help='The directory to save data like model checkpoints, loss, and images to.', type=pathlib.Path)
parser.add_argument('-l', '--logfile', help='The file to log to.')
args = parser.parse_args()

# ----- logger -----

logging.basicConfig(filename=args.logfile, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(message)s')
logger = logging.getLogger(__name__)

logging.info(f'Program started.')

# ----- data dir -----

if args.data_dir.exists():
    logger.critical(f'{args.data_dir} already exists on your filesystem. Please specify an empty location.')  # TODO: can maybe say empty folder is ok.
    quit()
else:
    DATA_DIR = args.data_dir

# ----- device -----

import torch

if torch.cuda.is_available():
    logger.info(f'CUDA is available')
    DEVICE = torch.device('cuda')
else:
    logger.info(f'CUDA is not available; falling back to CPU.')
    DEVICE = torch.device('cpu')

logger.info(f'Set DEVICE={DEVICE}')

# ----- graph -----

'''
for now I'm going to assume monotonically increasing x values starting at 0.
later I can also save x values if need be.
'''

def graph(datapoint, graph_name, data_dir=DATA_DIR):
    # so we need an np array of x values, and of y values.
    x_values, y_values = np.save()
    pass
