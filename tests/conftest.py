import pytest
from PIL import Image
import librosa
import os
import subprocess
from pathlib import Path

import numpy as np
from nibabel.testing import data_path
from nii_reader import NiiReader
import nibabel as nib

data_dir = (Path(__file__).parent / 'toy_data').absolute()


@pytest.fixture(scope='function')
def image_fn():
    return str(data_dir / 'example4d.nii.gz')


@pytest.fixture(scope='function')
def expected_image():
    example_file = os.path.join(data_path, 'example4d.nii.gz')
    img = nib.load(example_file)
    return img


@pytest.fixture(scope='function')
def niiReader():
    return NiiReader()
