import numpy as np
from jina import Document, DocumentArray, Flow

from nii_reader import NiiReader


def test_spell_check_integration(expected_image, tmpdir):
    input_docs = DocumentArray(
        [Document(id='ex4', uri='tests/toy_data/example4d.nii.gz')]
    )
    with Flow().add(name='nii', uses=NiiReader) as f:
        results = f.post(on='/', inputs=input_docs, return_results=True)

        for doc in results[0].docs:
            np.testing.assert_allclose(doc.blob.shape, expected_image.shape)
            np.testing.assert_allclose(doc.blob, expected_image.get_fdata())
