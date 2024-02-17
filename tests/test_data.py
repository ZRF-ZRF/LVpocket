import os
import tempfile

import numpy as np
import h5py

from tfbio.data import Featurizer

import pytest

from kalasanty.data import print_progress, prepare_dataset, DataWrapper


path = os.path.dirname(os.path.abspath(__file__))

test_prepared_dataset = os.path.join(path, 'test_data.hdf')
dataset_prot_f = 18
dataset_pocket_f = 3
dataset_size = 10
dataset_box = 36

pdbbind_dataset_path = os.path.join(path, 'datasets', 'pdbbind')
scpdb_dataset_path = os.path.join(path, 'datasets', 'scpdb')

prot_featurizer = Featurizer(save_molecule_codes=False)
pocket_featurizer = Featurizer(atom_codes={}, named_properties=(),
                               save_molecule_codes=False,
                               smarts_properties=['*'],
                               smarts_labels=['any_atom'])


@pytest.mark.parametrize('num_steps', (1, 3, 10),
                         ids=lambda x: '%s steps' % x)
def test_print_progress(capsys, num_steps):

    for i in print_progress(range(100), num_steps=num_steps):
        pass
    out, err = capsys.readouterr()
    printed_str = ''.join('%s elements processed\n' % i
                          for i in range(0, 100, num_steps))
    assert out == printed_str


@pytest.mark.parametrize('ids', (None, 'all', 'half'),
                         ids=('no IDs', 'all IDs', 'half of IDs'))
@pytest.mark.parametrize(
    'dataset_path, db_format',
    ((scpdb_dataset_path, 'scpdb', ),
     (pdbbind_dataset_path, 'pdbbind')),
    ids=('scpdb', 'pdbbid'))
def test_prepare_dataset(ids, dataset_path, db_format):
    keys = ('coords', 'features', 'centroid',
            'pocket_coords', 'pocket_features')
    num_prot_f = len(prot_featurizer.FEATURE_NAMES)
    num_pocket_f = len(pocket_featurizer.FEATURE_NAMES)
    all_ids = list(os.listdir(dataset_path))

    if ids == 'all':
        ids = all_ids
    elif ids == 'half':
        ids = all_ids[::2]

    with tempfile.NamedTemporaryFile(suffix='.hdf') as tmp:
        prepare_dataset(dataset_path, prot_featurizer, pocket_featurizer,
                        ids=ids, hdf_path=tmp.name, hdf_mode='w', db_format=db_format)
        if ids is None:
            ids = all_ids
        stored_ids = {i[:4] for i in ids}

        with h5py.File(tmp.name, 'r') as f:
            for idx in stored_ids:
                assert idx in f
                for key in keys:
                    assert key in f[idx]
                assert f[idx]['features'].shape[1] == num_prot_f
                assert f[idx]['pocket_features'].shape[1] == num_pocket_f
                assert len(f[idx]['features']) == len(f[idx]['coords'])
                assert (len(f[idx]['pocket_features'])
                        == len(f[idx]['pocket_coords']))


def test_prepare_dataset_printing(capsys):
    with tempfile.NamedTemporaryFile(suffix='.hdf') as tmp:
        prepare_dataset(scpdb_dataset_path, prot_featurizer, pocket_featurizer,
                        hdf_path=tmp.name, hdf_mode='w', db_format='scpdb',
                        progress_bar=lambda x: print_progress(x, num_steps=2))
    out, err = capsys.readouterr()
    printed_str = ''.join('%s elements processed\n' % i
                          for i in [0, 2, 4])
    assert out == printed_str


@pytest.mark.parametrize('test_set_fraction', (
    0.2,
    0.45,
), ids=lambda x: '%i%% in test set' % (100 * x))
def test_no_ids(test_set_fraction):
    test_set_size = int(dataset_size * test_set_fraction)

    with DataWrapper(test_prepared_dataset, test_set=test_set_fraction) as data:
        assert ~data.closed
        assert isinstance(data.data_handle, h5py.File)
        assert data.pdbids == list(data)
        assert data.x_channels == dataset_prot_f
        assert data.y_channels == dataset_pocket_f
        assert len(data.test_set) == test_set_size
        assert len(data.training_set) == dataset_size - test_set_size
        assert '10gs' in data

        x, y = data.prepare_complex('10gs')
        num_atoms = len(data['10gs']['coords'])
        assert x.shape == tuple([1] + [dataset_box] * 3 + [dataset_prot_f])
        assert y.shape == tuple([1] + [dataset_box] * 3 + [dataset_pocket_f])
        assert x[..., :9].sum() == num_atoms
        assert y.sum() == 110
    assert data.closed


def test_no_test_set():
    with pytest.raises(ValueError, match='should be between 0 and 1'):
        DataWrapper(test_prepared_dataset, test_set=0.0)

    with pytest.raises(ValueError, match='should be between 0 and 1'):
        DataWrapper(test_prepared_dataset, test_set=1.0)

    with pytest.raises(TypeError, match='list of IDs'):
        DataWrapper(test_prepared_dataset, test_set='10gs')

    with DataWrapper(test_prepared_dataset, load_data=True) as data:
        assert ~data.closed
        assert isinstance(data.data_handle, dict)
        assert len(data.test_set) == 0
        assert len(data) == len(data.pdbids)
        assert data.training_set == data.pdbids
    assert data.closed


@pytest.mark.parametrize('footprint, err, message', (
    (-3, ValueError, 'negative'),
    (100, ValueError, 'bigger than box'),
    (np.ones((3, 3, 3)), ValueError, 'shape'),
    (0.5, TypeError, 'should be either int or np.ndarray'),
), ids=('negative', 'too big', 'wrong shape',
        'wrong type'))
def test_wrong_footprint(footprint, err, message):
    with pytest.raises(err, match=message):
        DataWrapper(test_prepared_dataset, footprint=footprint)


@pytest.mark.parametrize('max_dist,scale', ((11., 1.), (9., 2.), (30., 0.2)),
                         ids=('scale 1:1', 'scale 2:1', 'scale 1:5'))
@pytest.mark.parametrize('footprint', (
    np.ones((1, 3, 3, 3, 1)),
    0,
    3,
), ids=('array footprint', '0 footprint', 'int footprint'))
def test_sample_generator(max_dist, scale, footprint):
    from math import ceil

    train_pdbids = ['16pk', '184l', '185l', '186l', '187l', '188l', '1a07']
    test_pdbids = ['10gs', '11gs', '13gs']
    pdbids = train_pdbids + test_pdbids

    data = DataWrapper(test_prepared_dataset, pdbids=pdbids,
                       test_set=test_pdbids, load_data=True,
                       footprint=footprint, max_dist=max_dist, scale=scale,
                       max_translation=1)

    box = ceil(2. * max_dist / (1. / scale) + 1)
    x_shape = tuple([1] + [box] * 3 + [dataset_prot_f])
    y_shape = tuple([1] + [box] * 3 + [dataset_pocket_f])
    assert ~data.closed
    assert isinstance(data.data_handle, dict)

    generator = data.sample_generator('all', random_order=False)

    for i in range(3 * len(pdbids)):
        k, x, y = next(generator)
        assert x.shape == x_shape
        assert y.shape == y_shape
        assert k == pdbids[i % len(pdbids)]

    generator = data.sample_generator()
    order = [-1] * len(train_pdbids)

    for i in range(len(train_pdbids)):
        k, x, y = next(generator)
        assert x.shape == x_shape
        assert y.shape == y_shape
        assert k in train_pdbids
        idx = train_pdbids.index(k)
        assert order[idx] == -1
        order[idx] = i

    assert order != list(range(len(train_pdbids)))

    generator = data.sample_generator('test', transform=False,
                                      random_order=False)
    results = [next(generator) for _ in range(len(test_pdbids))]

    for i in range(len(test_pdbids)):
        for prev, current in zip(results[i], next(generator)):
            assert np.all(prev == current)

    generator = data.sample_generator('test', random_order=False)
    order = [-1] * len(test_pdbids)
    for i in range(len(test_pdbids)):
        k, x, y = next(generator)
        idx = test_pdbids.index(k)
        assert order[idx] == -1
        order[idx] = i
        assert (x != 0).any()
        assert (y != 0).any()
        assert ((x != results[i][1]).any())
        assert ((y != results[i][2]).any())
    assert order == list(range(len(test_pdbids)))

    data.close()
    assert data.closed


def test_closed_wrapper():
    with DataWrapper(test_prepared_dataset) as data:
        pass
    with pytest.raises(RuntimeError):
        len(data)
    with pytest.raises(RuntimeError):
        data['10gs']
    with pytest.raises(RuntimeError):
        '10gs' in data

    with pytest.raises(RuntimeError):
        next(data.sample_generator())
    with pytest.raises(RuntimeError):
        next(data.batch_generator())
    with pytest.raises(RuntimeError):
        data.prepare_complex('10gs')
