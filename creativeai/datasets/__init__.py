"""
A bunch of data loading procedures.
"""

import os
import pandas as ps

script_folder = os.path.dirname(os.path.realpath(__file__))

def read_imdb():
    """
    Reads imdb dataset.

    Returns
    -------
    dataset: A set of customer feedback texts
    """

    dataset_file = os.path.join(script_folder, 'labeledTrainData.tsv')

    # check if file exists
    if not os.path.exists(dataset_file):
        raise BaseException('Please download the IMDB dataset first. '
                            'See `datasets` folder for further info')

    # load the csv data
    data = ps.read_csv(dataset_file,  sep='\t')

    return data