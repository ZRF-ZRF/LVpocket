import os
from warnings import warn
import re

from random import shuffle, choice, sample

import numpy as np
from scipy import ndimage
import h5py

import pybel

import tfbio.net
import tfbio.data
from skimage.draw import ellipsoid


__all__ = [
    'print_progress',
    'prepare_dataset',
    'DataWrapper',
]


def print_progress(iterable, num_steps=10):
    """Report progress while passing through an `iterable` by printing number of
    processed elements every `num_steps` steps. This function yields elements
    from the original iterable object, so this:

    >>> for i in iterable:
    ...     <some action>

    can be changed to this:

    >>> for i in print_progress(iterable):
    ...     <some action>
    """

    for num, i in enumerate(iterable):
        yield i
        if num % num_steps == 0:
            print('%s elements processed' % num)


def _pdbbind_paths(data_path, idx):
    return ('%s/%s/%s_ligand.mol2' % (data_path, idx, idx),
            '%s/%s/%s_protein.mol2' % (data_path, idx, idx))


def _scpdb_paths(data_path, idx):
    return ('%s/%s/cavity6.mol2' % (data_path, idx),
            '%s/%s/protein.mol2' % (data_path, idx))


def _get_binary_features(mol):
    coords = []

    for a in mol.atoms:
        coords.append(a.coords)
    coords = np.array(coords)
    features = np.ones((len(coords), 1))
    return coords, features


def prepare_dataset(data_path, protein_featurizer, pocket_featurizer=None,
                    ids=None, hdf_path='pockets.hdf', hdf_mode='w',
                    progress_bar=None, db_format='scpdb', verbose=False):
    """Compute features for proteins and pockets and save results in HDF file.

    Parameters
    ----------
    data_path : str
        Path to the directory with structures. For now only mol2 format is
        supported. The directory should be organized as PDBbind or sc-PDB database
        (see `db_format`):
    protein_featurizer, pocket_featurizer: tfbio.data.Featurizer objects
        Featurizers to prepare protein and pocket. If pocket_featurizer is not
        specified, single binary presence/absence feature will be used.
    ids: list of strings, optional (default=None)
        List of complexes to prepare. If not specified, all complexes in the
        directory will be used.
    hdf_path: str, optional (default='pockets.hdf')
        Path to output file
    hdf_mode: str, optional (default='w')
        Mode in which hdf_path file should be opened (passed to `h5py`).
    progress_bar: callable, optional (default=None)
        A function that prints progress bar while looping over iterable.
        You can for example use `tqdm` or `tqdm_notebook` from `tqdm` package,
        or `print_progress` function defined in this module.
    db_format: str, optional ('pdbbind' or 'scpdb', default='scpdb')
        There are two types of databases supported: sc-PDB-like database (with
        cavities) and PDBbind-like (with ligands). If 'scpdb' is selected, data
        directory should be organized as follows:
            - data_path
              - pdbid1_1
                - cavity6.mol2
                - protein.mol2
              - pdbid1_2
                - cavity6.mol2
                - protein.mol2
              - pdbid2_1
                - cavity6.mol2
                - protein.mol2
        All pockets for the same pdbid (here: 'pdbid1_1' and 'pdbid1_2') will
        be stored together.
        If 'pdbbind' is selected, data directory should be organized as follows:
            - data_path
              - pdbid1
                - pdbid1_ligand.mol2
                - pdbid1_protein.mol2
              - pdbid2
                - pdbid2_ligand.mol2
                - pdbid2_protein.mol2
        Both proteins and ligands are required (ligands will be used to define
        pockets).
    verbose: bool, optional (default=False)
        Whether to print messages about dataset.
    """

    if pocket_featurizer is None:
        featurize_pocket = _get_binary_features
    else:
        featurize_pocket = pocket_featurizer.get_features

    # just iterate over ids if progress bar method is not specified
    if progress_bar is None:
        progress_bar = iter

    if db_format == 'scpdb':
        get_paths = _scpdb_paths
        get_id = lambda structure_id: re.sub('_[0-9]+$', '', structure_id)
    elif db_format == 'pdbbind':
        get_paths = _pdbbind_paths
        get_id = lambda structure_id: structure_id
    else:
        raise ValueError('Unrecognised db_format "{}"'.format(db_format))

    # TODO: save feature names in metadata
    # TODO: allow other mol formats
    data_path = os.path.abspath(data_path)
    if ids is None:
        ids = os.listdir(data_path)

    multiple_pockets = {}

    with h5py.File(hdf_path, mode=hdf_mode) as f:
        for structure_id in progress_bar(ids):
            pocket_path, protein_path = get_paths(data_path, structure_id)
            pocket = next(pybel.readfile('mol2', pocket_path))
            protein = next(pybel.readfile('mol2', protein_path))

            pocket_coords, pocket_features = featurize_pocket(pocket)
            prot_coords, prot_features = protein_featurizer.get_features(protein)

            centroid = prot_coords.mean(axis=0)
            pocket_coords -= centroid
            prot_coords -= centroid

            group_id = get_id(structure_id)
            if group_id in f:
                group = f[group_id]
                if not np.allclose(centroid, group['centroid'][:], atol=0.5):
                    warn('Structures for %s are not aligned, ignoring pocket %s' % (group_id, structure_id))
                    continue

                # another pockets from same structure - extend pockets' data
                multiple_pockets[group_id] = multiple_pockets.get(group_id, 1) + 1

                for key, data in (('pocket_coords', pocket_coords),
                                  ('pocket_features', pocket_features)):
                    data = np.concatenate((group[key][:], data))
                    del group[key]
                    group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')
            else:
                # first pocket for the structure - create all datasets
                group = f.create_group(group_id)
                for key, data in (('coords', prot_coords),
                                  ('features', prot_features),
                                  ('pocket_coords', pocket_coords),
                                  ('pocket_features', pocket_features),
                                  ('centroid', centroid)):
                    group.create_dataset(key, data=data, shape=data.shape, dtype='float32', compression='lzf')

    if verbose and len(multiple_pockets) > 0:
        print('Found multiple pockets for:')
        for idx, num in multiple_pockets.items():
            print('{idx} ({num})'.format(idx=idx, num=num))


def get_box_size(scale, max_dist):
    return int(np.ceil(2 * max_dist * scale + 1))


class DataWrapper():
    """Wraps dataset saved in HDF file.

    Attributes
    ----------
    data_handle: dict or h5py.File
        Dataset handle
    keys: tuple of strings
        attributes of a sample in a dataset, i.e.:
          * 'coords': protein coordinates, shape (N, 3)
          * 'features': protein features, shape (N, K)
          * 'centroid': centroid of original complex
          * 'pocket_coords': pocket coordinates, shape (M, 3)
          * 'pocket_features': pocket features, shape (M, L)
    pdbids, training_set, test_set: lists of strings
        Lists with all, training set, and test set IDs, respectively.
    box_size: int
        Size of a box surrounding the protein
    x_channels: int
        Number of features describing the protein
    y_channels: int
        Number of features describing the pocket
    """

    def __init__(self, hdf_path, pdbids=None, test_set=None, load_data=False,
                 max_dist=35, scale=0.5, footprint=None, max_translation=5):
        """Creates the wrapper

        Parameters
        ----------
        hdf_path: str
            Path to the dataset (can be created with `prepare_dataset` function)
        pdbids: list of strings, optional (default=None)
            List of complexes to use. If not specified, all complexes in the
            dataset will be used.
        test_set: float or list of strings, optional (default=None)
            Test set can be either defined with list of complexes (must be
            included in pdbids), or fraction of dataset to use as a test set.
            All other complexes will be used as training set.
        load_data: bool, optional (default=False)
            If true, dataset will be loaded into the memory and stored as
            dictionary. Otherwise opened h5py.File object is used.
        max_dist: float, optional (default=35)
            Atoms with coordinates more than `max_dist` away from a center will
            be ignored.
        scale: float, optional (default=0.5)
            Structure scaling factor.
        footprint: int or np.ndarray, shape (1, N, M, L, 1), optional (default=None)
            Margin used to define the pocket based on ligand structure. If not
            specified sphere with radius=2 is used.
        max_translation: float, optional (default=5)
            Maximum translation to use (in each direction) in data augmentation.
        """

        self.hdf_path = os.path.abspath(hdf_path)

        self.keys = ('coords', 'features', 'centroid',
                     'pocket_coords', 'pocket_features')
        self.load_data = load_data
        self.pdbids = pdbids
        self.data_handle = None
        self._open_data_handle()

        if test_set is not None:
            if isinstance(test_set, (set, tuple, list, np.ndarray)):
                self.test_set = test_set
            elif isinstance(test_set, float):
                if not (0 < test_set < 1):
                    raise ValueError('test_set should be between 0 and 1 '
                                     '(exclusive), got %s instead' % test_set)
                num_test = int(len(self.pdbids) * test_set)
                self.test_set = sample(self.pdbids, num_test)
            else:
                raise TypeError('test_set can be either specified with list of'
                                ' IDs or a fraction of the data (float between'
                                ' 0 and 1, exclusive), got %s instead'
                                % type(test_set))
        else:
            self.test_set = []
        # TODO optimize
        self.training_set = [pdbid for pdbid in self.pdbids
                             if pdbid not in self.test_set]

        self.max_translation = max_translation
        self.max_dist = max_dist
        self.scale = scale
        self.box_size = get_box_size(scale, max_dist)

        if footprint is not None:
            if isinstance(footprint, int):
                if footprint == 0:
                    footprint = np.ones([1] * 5)
                elif footprint < 0:
                    raise ValueError('footprint cannot be negative')
                elif (2 * footprint + 3) > self.box_size:
                    raise ValueError('footprint cannot be bigger than box')
                else:
                    footprint = ellipsoid(footprint, footprint, footprint)
                    footprint = footprint.reshape((1, *footprint.shape, 1))
            elif isinstance(footprint, np.ndarray):
                if not ((footprint.ndim == 5) and (len(footprint) == 1)
                        and (footprint.shape[-1] == 1)):
                    raise ValueError('footprint shape should be '
                                     '(1, N, M, L, 1), got %s instead'
                                     % str(footprint.shape))
            else:
                raise TypeError('footprint should be either int or np.ndarray '
                                'of shape (1, N, M, L, 1), got %s instead'
                                % type(footprint))
            self.footprint = footprint
        else:
            footprint = ellipsoid(2, 2, 2)
            self.footprint = footprint.reshape((1, *footprint.shape, 1))

        pdbid = self.pdbids[0]
        self.x_channels = self.data_handle[pdbid]['features'].shape[1]
        self.y_channels = self.data_handle[pdbid]['pocket_features'].shape[1]

    def __enter__(self):
        self._open_data_handle()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open_data_handle(self):
        if hasattr(self, 'closed') and not self.closed:
            # it's open
            return

        if self.load_data:
            if self.data_handle is None:
                self.data_handle = {}
                with h5py.File(self.hdf_path, mode='r') as f:
                    if self.pdbids is None:
                        self.pdbids = list(f.keys())
                    for pid in self.pdbids:
                        self.data_handle[pid] = {}
                        for key in self.keys:
                            self.data_handle[pid][key] = f[pid][key][:]
        else:
            self.data_handle = h5py.File(self.hdf_path, mode='r')
            if self.pdbids is None:
                self.pdbids = list(self.data_handle.keys())
        self.closed = False

    def close(self):
        if isinstance(self.data_handle, h5py.File) and self.data_handle.id:
            self.data_handle.close()
        self.closed = True

    def prepare_complex(self, pdbid, rotation=0, translation=(0, 0, 0),
                        vmin=0, vmax=1):
        """Prepare complex with given pdbid.

        Parameters
        ----------
        pdbid: str
            ID of a complex to prepare
        rotation: int or np.ndarray (shape (3, 3)), optional (default=0)
            Rotation to apply. It can be either rotation matrix or ID of
            rotatation defined in `tfbio.data` (0-23)
        translation: tuple of 3 floats, optional (default=(0, 0, 0))
            Translation to apply
        vmin, vmax: floats, optional (default 0 and 1)
            Clip values generated for pocket to this range

        Returns
        -------
        rec_grid: np.ndarray
            Grid representing protein
        pocket_dens: np.ndarray
            Grid representing pocket
        """
        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')

        resolution = 1. / self.scale
        structure = self.data_handle[pdbid]
        rec_coords = tfbio.data.rotate(structure['coords'][:], rotation)
        rec_coords += translation
        rec_grid = tfbio.data.make_grid(rec_coords, structure['features'][:],
                                        max_dist=self.max_dist,
                                        grid_resolution=resolution)

        pocket_coords = tfbio.data.rotate(structure['pocket_coords'][:],
                                          rotation)
        pocket_coords += translation
        pocket_dens = tfbio.data.make_grid(pocket_coords,
                                           structure['pocket_features'][:],
                                           max_dist=self.max_dist)
        margin = ndimage.maximum_filter(pocket_dens,
                                        footprint=self.footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i],
                                             zoom)
                                for i in range(self.y_channels)], -1)
        pocket_dens = np.expand_dims(pocket_dens, 0)

        return rec_grid, pocket_dens

    def __getitem__(self, pdbid):
        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')
        return self.data_handle[pdbid]

    def __contains__(self, pdbid):
        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')
        return (str(pdbid) in self.data_handle)

    def __iter__(self):
        return iter(self.pdbids)

    def __len__(self):
        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')
        return len(self.data_handle)

    def sample_generator(self, subset='training', transform=True,
                         random_order=True):
        """Yields samples from a given subset ('training', 'test' or 'all').
        By default complexes are randomly transformed (rotated and translated)
        and randomly ordered."""

        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')

        if subset == 'training':
            pdbids = self.training_set[:]
        elif subset == 'test':
            pdbids = self.test_set[:]
        elif subset == 'all':
            pdbids = self.pdbids[:]

        while True:
            if random_order:
                shuffle(pdbids)
            for k in pdbids:
                if transform:
                    rot = choice(range(24))
                    tr = self.max_translation * np.random.rand(1, 3)
                else:
                    rot = 0
                    tr = (0, 0, 0)
                r, p = self.prepare_complex(k, rotation=rot, translation=tr)
                yield (k, r, p)

    def batch_generator(self, batch_size=5, **kwargs):
        """Yields batches of samples. All arguments except batch_size are
        passed to the `sample_generator` method.
        """

        if self.closed:
            raise RuntimeError('Trying to use closed DataWrapper')

        examples = self.sample_generator(**kwargs)
        while True:
            receptors = []
            pockets = []
            for _ in range(batch_size):
                _, receptor, pocket = next(examples)
                receptors.append(receptor)
                pockets.append(pocket)
            yield np.vstack(receptors), np.vstack(pockets)

