from parse import get_arguments
from baseline_training import Baseline
from architectures.baseline_v2 import SingleBlock

if __name__ == "__main__":
    opt = get_arguments()
    baseline = Baseline(SingleBlock, opt)
    baseline.train()