__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from nii_reader import NiiReader


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.dtype == np.float32


def test_no_documents(nii_reader: NiiReader):
    docs = DocumentArray()
    nii_reader.load(docs=docs)
    assert len(docs) == 0  # SUCCESS


def test_docs_no_uris(nii_reader: NiiReader):
    docs = DocumentArray([Document()])
    nii_reader.load(docs=docs)
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_extract(expected_image, image_fn, nii_reader: NiiReader, batch_size: int):
    docs = DocumentArray([Document(uri=image_fn) for _ in range(batch_size)])
    nii_reader.load(docs=docs)
    for doc in docs:
        np.testing.assert_allclose(doc.blob.shape, expected_image.shape)
        np.testing.assert_allclose(doc.blob, expected_image.get_fdata())


def test_load_with_datauri(expected_image, image_fn, nii_reader: NiiReader):
    doc = Document(uri=image_fn)
    doc.convert_uri_to_datauri()
    docs = DocumentArray([doc])
    nii_reader.load(docs=docs)

    for doc in docs:
        np.testing.assert_allclose(doc.blob.shape, expected_image.shape)
        np.testing.assert_allclose(doc.blob, expected_image.get_fdata())


def test_catch_wrong_format(caplog, nii_reader: NiiReader):
    docs = DocumentArray(
        [Document(uri='tests/toy_data/example4d.ni.gz')]
    )  # wrong nifti format
    nii_reader.logger.propagate = True
    nii_reader.load(docs=docs)
    assert 'Cannot work out file' in caplog.text


def test_catch_no_file(caplog, nii_reader: NiiReader):
    docs = DocumentArray(
        [Document(uri='tests/toy_data/example5d.ni.gz')]
    )  # no such file exists
    nii_reader.logger.propagate = True
    nii_reader.load(docs=docs)
    assert 'No such file' in caplog.text


def test_dtype_string(expected_image, image_fn):
    docs = DocumentArray([Document(uri=image_fn)])
    niiReader = NiiReader(dtype=None)
    niiReader.load(docs=docs)
    for doc in docs:
        np.testing.assert_allclose(doc.blob.shape, expected_image.shape)
        np.testing.assert_allclose(doc.blob, expected_image.get_fdata())
