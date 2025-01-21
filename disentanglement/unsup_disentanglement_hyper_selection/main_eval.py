import argparse
import logging
import sys
import os
from configparser import ConfigParser

from torch import optim

from models.vae import VAE, Encoder, Decoder
from models.regression import WTPregression
from training.training import Trainer
from training.evaluate import Evaluator
from models.modelIO import save_model, load_model, load_metadata
from models.losses import BtcvaeLoss
from dataset.datasets import get_dataloaders, get_img_size
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,get_config_section)
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from torchsummary import summary

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom']
EXPERIMENTS = ADDITIONAL_EXP

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,help="Name of the model for storing and loading purposes.", default=default_config['name'])
    parser.add_argument('--btcvae-A', type=float,default=default_config['btcvae_A'],help="Weight of the MI term (alpha in the paper).")
    parser.add_argument('--btcvae-B', type=float,default=default_config['btcvae_B'],help="Weight of the TC term (beta in the paper).")
    parser.add_argument('--btcvae-G', type=float,default=default_config['btcvae_G'],help="Weight of the dim-wise KL term (gamma in the paper).")
    parser.add_argument('--btcvae-M', type=float,default=default_config['btcvae_M'],help="Weight of the MSE term (delta in the paper).")
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-e', '--epochs', type=int,default=default_config['epochs'],help='Maximum number of epochs to run for.')
    parser.add_argument('-mt', '--model-type', type=str,default=default_config['model_type'],help='Type of Model')
    parser.add_argument('-a', '--reg-anneal', type=float,default=default_config['reg_anneal'],help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")
    parser.add_argument('-tv', '--threshold-val', type=float,default=default_config['threshold_val'],help='Threshold for Masking.')
    parser.add_argument('--sup_signal1', type=str,default=default_config['sup_signal1'],help="Choice of Signal")
    parser.add_argument('--sup_signal2', type=str,default=default_config['sup_signal2'],help="Choice of Signal")
    parser.add_argument('--sup_signal3', type=str,default=default_config['sup_signal3'],help="Choice of Signal")
    parser.add_argument('--sup_signal4', type=str,default=default_config['sup_signal4'],help="Choice of Signal")
    parser.add_argument('--sup_signal5', type=str,default=default_config['sup_signal5'],help="Choice of Signal")
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],help='List of indices to of images to put at the begining of the samples.')
    args = parser.parse_args()
    return args


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")
    logger = logging.getLogger(__name__)

    set_seed(int(args.seed))
    device = get_device(is_gpu=not default_config['no_cuda'])
    
    experiment_name = args.name
    model_dir = os.path.join(RES_DIR, experiment_name)
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir, model_type=args.model_type,threshold_val=args.threshold_val,sup_signal1=args.sup_signal1,sup_signal2=args.sup_signal2,sup_signal3=args.sup_signal3,sup_signal4=args.sup_signal4,sup_signal5=args.sup_signal5)
    model.eval()  # don't sample from latent: use mean
    dataset = "cars"
    train_loader, validation_loader, train_loader_unshuffled, train_loader_batch1, test_loader = get_dataloaders("cars",batch_size=int(default_config['batch_size']),eval_batchsize=int(default_config['eval_batchsize']),model_name=args.name,sup_signal1=args.sup_signal1,sup_signal2=args.sup_signal2,sup_signal3=args.sup_signal3,sup_signal4=args.sup_signal4,sup_signal5=args.sup_signal5,logger=logger)

    loss_f = BtcvaeLoss(rec_dist=default_config['rec_dist'],steps_anneal=float(args.reg_anneal),n_data=len(test_loader.dataset),alpha=float(args.btcvae_A),beta=float(args.btcvae_B),gamma=float(args.btcvae_G),delta=float(args.btcvae_M),sup_signal1=args.sup_signal1,sup_signal2=args.sup_signal2,sup_signal3=args.sup_signal3,sup_signal4=args.sup_signal4,sup_signal5=args.sup_signal5)
    evaluator = Evaluator(model, loss_f,device=device,logger=logger,save_dir=model_dir,experiment_name=args.name,model_type=args.model_type,is_progress_bar=not default_config['no_progress_bar'],file_type="test")
    evaluator(test_loader)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
