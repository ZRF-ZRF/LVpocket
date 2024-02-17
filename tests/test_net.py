import os

import numpy as np
import h5py

import tempfile

import pytest

from keras import backend as K
from keras.layers import Input, Convolution3D, concatenate
from keras.models import Model
from keras.optimizers import Adam

import pybel

from tfbio.data import Featurizer

from kalasanty.net import dice_np, dice, dice_loss, ovl_np, ovl, ovl_loss, DataWrapper, UNet


path = os.path.dirname(os.path.realpath(__file__))
test_dataset = os.path.join(path, 'test_data.hdf')
protein_file = os.path.join(path, 'datasets', 'scpdb', '2qfo_1', 'protein.mol2')
featurizer = Featurizer(save_molecule_codes=False)
num_features = len(featurizer.FEATURE_NAMES)

input_shape = (1, 4, 2, 3, 1)
arr_zeros = np.zeros(input_shape)
arr_ones = np.ones(input_shape)


def teardown_function(function):
    K.clear_session()


@pytest.fixture(scope='function')
def data():
    data = DataWrapper(test_dataset, test_set=0.2, max_dist=52, scale=0.33)
    yield data
    data.close()


@pytest.mark.parametrize('smoothing', (0, 0.1, 0.001),
                         ids=lambda x: 'smoothing %s' % x)
def test_dice(smoothing):
    x = Input(input_shape[1:])
    m = Model(inputs=x, outputs=x)

    arr_random = np.random.choice([0, 1], size=input_shape,
                                  p=[0.75, 0.25])
    arrays = (arr_random, arr_zeros, arr_ones)
    arr_sum = arr_random.sum()
    ones_sum = arr_ones.sum()

    scores = (1.0, smoothing / (arr_sum + smoothing),
              (2 * arr_sum + smoothing) / (arr_sum + ones_sum + smoothing))
    m.compile(Adam(), lambda x, y: dice(x, y, smoothing_factor=smoothing))

    for array, score in zip(arrays, scores):
        score_keras = m.evaluate(arr_random, array, verbose=0)
        score_np = dice_np(arr_random, array, smoothing_factor=smoothing)
        assert np.allclose(score_keras, score_np, 6)
        assert np.allclose(score_keras, score, 6)


@pytest.mark.parametrize('smoothing', (0, 0.1, 0.001),
                         ids=lambda x: 'smoothing %s' % x)
def test_ovl(smoothing):
    x = Input(input_shape[1:])
    m = Model(inputs=x, outputs=x)

    arr_random = np.random.choice([0, 1], size=input_shape,
                                  p=[0.75, 0.25])
    arr_sum = arr_random.sum()
    ones_sum = arr_ones.sum()
    arrays = (arr_random, arr_zeros, arr_ones)
    scores = (1.0, smoothing / (arr_sum + smoothing),
              (arr_sum + smoothing) / (ones_sum + smoothing))
    m.compile(Adam(), lambda x, y: ovl(x, y, smoothing_factor=smoothing))

    for array, score in zip(arrays, scores):
        score_keras = m.evaluate(arr_random, array, verbose=0)
        score_np = ovl_np(arr_random, array, smoothing_factor=smoothing)
        assert np.allclose(score_keras, score_np, 6)
        assert np.allclose(score_keras, score, 6)


def test_unet_from_data_handle(data):
    with pytest.raises(ValueError, match='you must either provide'):
        UNet()

    with pytest.raises(TypeError, match='data_handle should be a DataWrapper'):
        UNet(data_handle='10gs')

    model = UNet(data_handle=data)
    assert model.data_handle == data
    assert model.scale == data.scale
    assert model.max_dist == data.max_dist
    assert len(model.inputs) == 1
    assert model.inputs[0].shape[-1] == data.x_channels
    assert len(model.outputs) == 1
    assert model.outputs[0].shape[-1] == data.y_channels


@pytest.mark.parametrize('box_size', (4, 16), ids=lambda x: 'box=%s' % x)
@pytest.mark.parametrize('i', (5, 1), ids=lambda x: 'i=%s' % x)
@pytest.mark.parametrize('o', (2, 1), ids=lambda x: 'o=%s' % x)
def test_unet_from_layers(box_size, i, o):
    inputs = Input([box_size] * 3 + [i])
    conv1 = Convolution3D(filters=3, kernel_size=1, activation='elu',
                          padding='same')(inputs)
    outputs = Convolution3D(filters=o, kernel_size=1, activation='sigmoid',
                            padding='same')(conv1)

    model = UNet(inputs=inputs, outputs=outputs, box_size=box_size,
                 input_channels=i, output_channels=o)
    assert hasattr(model, 'data_handle')
    assert model.data_handle is None

    with pytest.raises(ValueError, match='input should be 5D'):
        UNet(inputs=inputs[0], outputs=inputs)

    with pytest.raises(ValueError, match='output should be 5D'):
        UNet(inputs=inputs, outputs=outputs[1])

    with pytest.raises(ValueError, match='input and output shapes do not match'):
        UNet(inputs=inputs, outputs=concatenate([outputs, outputs], 1))


@pytest.mark.parametrize('box_size', (36, 144), ids=lambda x: 'box=%s' % x)
@pytest.mark.parametrize('o', (4, 2), ids=lambda x: 'o=%s' % x)
def test_unet_with_featurizer(box_size, o):
    f = Featurizer()
    i = len(f.FEATURE_NAMES)

    with pytest.raises(TypeError, match='should be a tfbio.data.Featurize'):
        UNet(box_size=box_size, input_channels=i, output_channels=o,
             scale=0.5, featurizer=1)

    model = UNet(box_size=box_size, input_channels=i, output_channels=o,
                 scale=0.5, featurizer=f)
    assert hasattr(model, 'data_handle')
    assert model.data_handle is None
    assert hasattr(model, 'featurizer')
    assert isinstance(model.featurizer, Featurizer)


@pytest.mark.parametrize('box_size', (8, 16), ids=lambda x: 'box=%s' % x)
@pytest.mark.parametrize('i_channels', ([5, 3], [2, 1, 1]),
                         ids=lambda x: 'i=' + ','.join([str(i) for i in x]))
@pytest.mark.parametrize('o_channels', ([3, 3], [2, 1, 4]),
                         ids=lambda x: 'o=' + ','.join([str(i) for i in x]))
def test_multiple_inputs_outputs(box_size, i_channels, o_channels):
    inputs = [Input([box_size] * 3 + [i]) for i in i_channels]
    conv1 = [Convolution3D(filters=3, kernel_size=1, activation='elu',
                           padding='same')(inp) for inp in inputs]
    conv1 = concatenate(conv1, axis=-1)
    outputs = [Convolution3D(filters=o, kernel_size=1, activation='sigmoid',
                             padding='same')(conv1) for o in o_channels]

    model = UNet(inputs=inputs, outputs=outputs, box_size=box_size,
                 input_channels=sum(i_channels),
                 output_channels=sum(o_channels))
    assert len(model.inputs) == len(i_channels)
    assert len(model.outputs) == len(o_channels)


@pytest.mark.parametrize('loss', (dice_loss, ovl_loss))
def test_training(data, loss):
    train_gen = data.batch_generator(batch_size=5)
    eval_gen = data.batch_generator(batch_size=5)
    test_gen = data.batch_generator(batch_size=2, subset='test')
    num_epochs = 2

    box_size = data.box_size
    input_channels = data.x_channels
    output_channels = data.y_channels

    inputs = Input((box_size, box_size, box_size, input_channels))
    outputs = Convolution3D(filters=output_channels, kernel_size=1,
                            activation='sigmoid')(inputs)

    model = UNet(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-6), loss=loss,
                  metrics=[dice, dice_loss, ovl, ovl_loss])
    model.fit_generator(train_gen, steps_per_epoch=2,
                        epochs=num_epochs, verbose=0)

    for scores in (model.evaluate_generator(eval_gen, steps=2),
                   model.evaluate_generator(test_gen, steps=1)):
        assert np.allclose(scores[1], -scores[2])
        assert np.allclose(scores[3], -scores[4])

    loss_change = model.history.history['loss']
    assert len(loss_change) == num_epochs
    assert (loss_change[0] != loss_change[1:]).all()


@pytest.mark.parametrize('kwargs, err', (
    ({'scale': 1.0}, ValueError),
    ({'max_dist': 35}, ValueError),
    ({'featurizer': 123}, TypeError),
    ({'featurizer': Featurizer()}, ValueError)
), ids=('wrong scale', 'wrong dist', 'wrong featurizer type',
        'wrong featurizer shape'))
@pytest.mark.parametrize('compiled', (True, False),
                         ids=('compiled', 'not compiled'))
@pytest.mark.filterwarnings('ignore:No training configuration found')
def test_load_wrong_args(data, kwargs, err, compiled):
    box_size = data.box_size
    i = data.x_channels
    o = data.y_channels

    model1 = UNet(box_size=box_size, input_channels=i,
                  output_channels=o, scale=data.scale,
                  data_handle=data)
    if compiled:
        model1.compile(optimizer=Adam(lr=1e-6),
                       loss='binary_crossentropy',
                       metrics=[dice, dice_loss, ovl, ovl_loss])

    with tempfile.NamedTemporaryFile(suffix='.hdf') as f:

        model1.save(f.name)

        with pytest.raises(err, match=list(kwargs)[0]):
            UNet.load_model(f.name, data_handle=data, **kwargs)


@pytest.mark.parametrize('kwargs', (
    {},
    {'max_dist': 52, 'scale': 0.33, 'featurizer': featurizer},
), ids=('no args', 'scale 1:3, dist=52, featurizer'))
@pytest.mark.parametrize('compiled', (True, False),
                         ids=('compiled', 'not compiled'))
@pytest.mark.filterwarnings('ignore:No training configuration found')
def test_save_load(data, kwargs, compiled):
    from keras.models import load_model as keras_load
    box_size = data.box_size
    i = data.x_channels
    o = data.y_channels

    model1 = UNet(box_size=box_size, input_channels=i,
                  output_channels=o, scale=data.scale,
                  data_handle=data)
    if compiled:
        model1.compile(optimizer=Adam(lr=1e-6),
                       loss='binary_crossentropy',
                       metrics=[dice, dice_loss, ovl, ovl_loss])
    weights1 = model1.get_weights()

    with tempfile.NamedTemporaryFile(suffix='.hdf') as f:

        model1.save(f.name)

        model2 = UNet.load_model(f.name, data_handle=data, **kwargs)
        weights2 = model2.get_weights()

        assert model1.to_json() == model2.to_json()
        for w1, w2 in zip(weights1, weights2):
            assert np.allclose(w1, w2)

    with tempfile.NamedTemporaryFile(suffix='.hdf') as f:
        model1.save_keras(f.name)

        model2 = keras_load(f.name)
        weights2 = model2.get_weights()

        for w1, w2 in zip(weights1, weights2):
            assert np.allclose(w1, w2)


@pytest.mark.parametrize('kwargs', (
    {'box_size': 30},
    {'input_channels': 1},
    {'output_channels': 4},
    {'scale': 2.0},
    {'featurizer': Featurizer()},
    {'inputs': Input([36] * 3 + [1])},
    {'outputs': Convolution3D(filters=3, kernel_size=1, activation='elu',
                              padding='same')(Input([36] * 3 + [1]))}
), ids=('box_size', 'input_channels', 'output_channels', 'scale', 'featurizer',
        'inputs, no outputs', 'outputs, no inputs'))
def test_incompatible_with_data_handle(data, kwargs):
    with pytest.raises(ValueError, match=list(kwargs)[0]):
        UNet(data_handle=data, **kwargs)


@pytest.mark.parametrize('input_shape, strides, message', (
    ([10] * 3 + [1], 1, 'input shape does not match box_size'),
    ([20] * 5 + [1], 1, 'input should be 5D'),
    ([20] * 3 + [1], 2, 'input and output shapes do not match'),
), ids=('box size', 'not 3D image', 'different shapes'))
def test_incompatible_layers_shapes(input_shape, strides, message):
    inputs = Input(input_shape)
    if message == 'input should be 5D':
        outputs = inputs
    else:
        outputs = Convolution3D(filters=3, kernel_size=1, activation='sigmoid',
                                padding='same', strides=strides)(inputs)

    with pytest.raises(ValueError, match=message):
        UNet(inputs=inputs, outputs=outputs, box_size=20)


@pytest.mark.parametrize('kwargs', (
    {'box_size': 30},
    {'input_channels': 1},
    {'output_channels': 4},
    {'featurizer': Featurizer()},
), ids=lambda x: ', '.join(str(k) for k in x))
def test_incompatible_with_layers(kwargs):
    inputs = Input([10] * 3 + [3])
    conv1 = Convolution3D(filters=3, kernel_size=1, activation='elu',
                          padding='same')(inputs)
    outputs = Convolution3D(filters=5, kernel_size=1, activation='sigmoid',
                            padding='same')(conv1)
    with pytest.raises(ValueError, match=list(kwargs)[0]):
        UNet(inputs=inputs, outputs=outputs, **kwargs)


def test_get_pockets_segmentation(data):
    with pytest.raises(ValueError, match='data_handle must be set'):
        model = UNet(box_size=data.box_size,
                     input_channels=data.x_channels,
                     output_channels=data.y_channels,
                     l2_lambda=1e-7)
        model.pocket_density_from_grid('10gs')

    with pytest.raises(ValueError, match='scale must be set'):
        model = UNet(box_size=data.box_size,
                     input_channels=data.x_channels,
                     output_channels=data.y_channels,
                     l2_lambda=1e-7, data_handle=data)
        model.scale = None
        model.pocket_density_from_grid('10gs')

    np.random.seed(42)
    model = UNet(box_size=data.box_size,
                 input_channels=data.x_channels,
                 output_channels=data.y_channels,
                 l2_lambda=1e-7, data_handle=data)
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy')
    density, *_ = model.pocket_density_from_grid('10gs')

    with pytest.raises(ValueError, match='not supported'):
        model.get_pockets_segmentation(np.array([density] * 2), 0.6)

    pocket = model.get_pockets_segmentation(density, 0.6)
    assert pocket.shape == (data.box_size,) * 3
    assert pocket.max() > 0
    assert len(np.unique(pocket)) - 1 <= pocket.max()


def test_save_pockets_cmap(data):
    model = UNet(data_handle=data, l2_lambda=1e-7)
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy')
    density, origin, step = model.pocket_density_from_grid('10gs')

    with pytest.raises(ValueError, match='saving more than one prediction'):
        model.save_density_as_cmap(np.concatenate((density, density)), origin,
                                   step)

    with tempfile.NamedTemporaryFile(suffix='.cmap') as cmap_file:
        fname = cmap_file.name
        model.save_density_as_cmap(density, origin, step, fname=fname)
        with h5py.File(fname, 'r') as f:
            assert 'Chimera' in f
            group = f['Chimera']
            assert len(group.keys()) == data.y_channels
            for i in range(data.y_channels):
                key = 'image%s' % (i + 1)
                assert key in group
                assert 'data_zyx' in group[key]
                dataset = group[key]['data_zyx'][:]
                assert np.allclose(density[0, ..., i].transpose([2, 1, 0]),
                                   dataset[:])


def test_save_pockets_cube(data):
    model = UNet(data_handle=data, l2_lambda=1e-7)
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy')
    density, origin, step = model.pocket_density_from_grid('10gs')

    with pytest.raises(ValueError, match='saving more than one prediction'):
        model.save_density_as_cube(np.concatenate((density, density)), origin,
                                   step)

    with pytest.raises(NotImplementedError, match='saving multichannel'):
        model.save_density_as_cube(density, origin, step)

    density = density[..., [0]]
    with tempfile.NamedTemporaryFile(suffix='.cube') as cmap_file:
        fname = cmap_file.name
        model.save_density_as_cube(density, origin, step, fname=fname)
        with open(fname, 'r') as f:
            # skip header
            for _ in range(7):
                f.readline()
            values = np.array(f.read().split()).reshape(density.shape)
            assert np.allclose(density, values.astype(float))


@pytest.mark.parametrize('box_size', (36, 72), ids=lambda x: 'box=%s' % x)
@pytest.mark.parametrize('o', (1, 3), ids=lambda x: 'o=%s' % x)
def test_predict_mol(box_size, o):
    mol = next(pybel.readfile('mol2', protein_file))
    with pytest.raises(ValueError, match='featurizer must be set'):
        model = UNet(box_size=box_size, scale=0.5, input_channels=num_features,
                     output_channels=o)
        model.pocket_density_from_mol(mol)

    with pytest.raises(ValueError, match='scale must be set'):
        model = UNet(featurizer=featurizer, box_size=box_size,
                     input_channels=num_features, output_channels=o)
        model.pocket_density_from_mol(mol)

    model = UNet(featurizer=featurizer, box_size=box_size, scale=0.5,
                 output_channels=o)
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy')

    with pytest.raises(TypeError, match='pybel.Molecule'):
        model.pocket_density_from_mol(protein_file)

    density, origin, step = model.pocket_density_from_mol(mol)
    assert (density > 0).any()


@pytest.mark.parametrize('box_size', (36, 72), ids=lambda x: 'box=%s' % x)
@pytest.mark.parametrize('o', (1, 2), ids=lambda x: 'o=%s' % x)
def test_predict_pocket_atoms(box_size, o):
    np.random.seed(42)
    mol = next(pybel.readfile('mol2', protein_file))

    model = UNet(featurizer=featurizer, box_size=box_size, scale=0.5,
                 output_channels=o)
    model.compile(optimizer=Adam(lr=1e-6), loss='binary_crossentropy')

    segmentation_kwargs = {'threshold': 0.55, 'min_size': 5}

    pocket_mols_atoms = model.predict_pocket_atoms(mol, dist_cutoff=3,
                                                   expand_residue=False,
                                                   **segmentation_kwargs)
    pocket_mols_residues = model.predict_pocket_atoms(mol, dist_cutoff=3,
                                                      expand_residue=True,
                                                      **segmentation_kwargs)
    assert len(pocket_mols_atoms) == len(pocket_mols_residues)
    assert len(pocket_mols_atoms) > 0
    for p1, p2 in zip(pocket_mols_atoms, pocket_mols_residues):
        assert isinstance(p1, pybel.Molecule)
        assert isinstance(p2, pybel.Molecule)
        assert len(p1.atoms) <= len(p2.atoms)
        res1 = set([res.idx for res in p1.residues])
        res2 = set([res.idx for res in p2.residues])
        assert res1 == res2
