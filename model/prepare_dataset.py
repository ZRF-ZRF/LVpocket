import argparse
import os


from kalasanty.data import prepare_dataset
from tfbio.data import Featurizer

from tqdm import tqdm


def input_path(path):
    """Check if input exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' % path)
    return path


def output_path(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', '-d', required=True, type=input_path,
                        help='path to the database')
    parser.add_argument('--include', '-i', type=input_path, nargs='*',
                        help='text file with IDs to use (each in separate line). '
                             'If not specified, all proteins in the database '
                             '(except those listed with --exclude) will be used. '
                             'Note that --exclude has higher priority (i.e. if '
                             'ID is specified with both -e and -i it will be skipped)')
    parser.add_argument('--exclude', '-e', type=input_path, nargs='*',
                        help='text file with IDs to skip (each in separate line). '
                             'It has higher priority than --include (i.e. if '
                             'ID is specified with both -e and -i it will be skipped)')
    parser.add_argument('--output', '-o', default='./pockets.hdf', type=output_path,
                        help='name for the file with the prepared structures')
    parser.add_argument('--mode', '-m', default='w',
                        type=str, choices=['r+', 'w', 'w-', 'x', 'a'],
                        help='mode for the output file (see h5py documentation)')
    parser.add_argument('--db_format', '-f', default='scpdb',
                        type=str, choices=['scpdb', 'pdbbind'],
                        help='way the database is structured - like sc-PDB or '
                             'like PDBbind (see examples in tests/datasets directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='whether to print messages')

    return parser.parse_args()


def main():
    args = parse_args()

    blacklist = []
    if args.exclude:
        for fname in args.exclude:
            with open(fname) as f:
                blacklist += f.read().split('\n')

    if args.include:
        all_ids = []
        for fname in args.include:
            with open(fname) as f:
                all_ids += list(filter(None, f.read().split('\n')))
    else:
        all_ids = os.listdir(args.dataset)

    ids = [i for i in all_ids if i not in blacklist]
    if len(ids) == 0:
        raise RuntimeError('No data to process (empty list of IDs)')

    protein_featurizer = Featurizer(save_molecule_codes=False)

    if args.verbose:
        print('%s IDs to process' % len(ids))
        print('(%s total, %s excluded)' % (len(all_ids), len(blacklist)))
        progress_bar = tqdm
    else:
        progress_bar = None

    prepare_dataset(args.dataset, protein_featurizer, ids=ids, db_format=args.db_format,
                    hdf_path=args.output, hdf_mode=args.mode,
                    progress_bar=progress_bar,  verbose=args.verbose)


if __name__ == '__main__':
    main()
