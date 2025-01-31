import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    if source[len(source) - 1] != '/': source = source + '/'
    if destination[len(destination) - 1] != '/': destination = destination + '/'

    print(source)

    file_list = os.listdir(source)
    random.shuffle(file_list)

    if (len(file_list) == 0): 
        print("No files in directory.")
        return
        
    threshold_train = int(0.9 * len(file_list))
    # threshold_val = int(0.9 * len(file_list))
    
    for i, file in enumerate(file_list):
        if i < threshold_train:
            os.rename(source + file, destination + 'train/' + file)
        else:
            os.rename(source + file, destination + 'val/' + file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)