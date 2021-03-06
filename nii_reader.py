__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import io
import os
import random
import string
import tempfile
import urllib.request
from pydoc import locate
from typing import Dict, Optional, Union

import nibabel.filebasedimages
import numpy as np
from jina import Executor, DocumentArray, requests
from jina.logging.logger import JinaLogger
from jina.types.document import _is_datauri
from monai.data.image_reader import NibabelReader


class NiiReader(Executor):
    """
    Load `NIfTI` format images based on `Nibabel` library.
    """

    def __init__(
        self,
        as_closest_canonical: bool = False,
        nibabel_args: Optional[Dict] = None,
        dtype: Union[np.dtype, str, type, None] = np.float32,
        *args,
        **kwargs,
    ):
        """
        :param as_closest_canonical: if True, load the image as closest to canonical axis format.
        :param nibabel_args: additional args for `nibabel.load` API. more details about available args:
            https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py
        :param dtype: dtype which the image will be converted to
        :param args: the *args for Executor
        :param kwargs: the **kwargs for Executor
        """
        super().__init__(*args, **kwargs)
        self.nibabel_args = nibabel_args or {}
        self.as_closest_canonical = as_closest_canonical or False
        self.dtype = None

        # when passed from yaml it is string
        if isinstance(dtype, str):
            actual_type = locate(dtype)
            if actual_type:
                self.dtype = actual_type
            else:
                raise RuntimeError(
                    f'Could not resolve type "{dtype}". '
                    f'Make sure you use "numpy.float32"-like syntax'
                )
        else:
            self.dtype = dtype

        self.reader = NibabelReader(
            self.as_closest_canonical, dtype=self.dtype, **self.nibabel_args
        )
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def load(self, docs: DocumentArray, **kwargs):
        """
        Load the NIfTI images available in the format `.nii` or `.nii.gz` from the Document.uri and store it in the `blob`
        of `Document` as an ndarray. Check out `https://nifti.nimh.nih.gov/nifti-2/` for the Data Format and shape
        :param docs: the input Documents with either the image file name or data URI in the `uri` field
        """
        for doc in docs:
            if doc.uri == '':
                self.logger.error(f'No uri passed for the Document: {doc.id}')
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                source_fn = (
                    self._save_uri_to_tmp_file(doc.uri, tmpdir)
                    if _is_datauri(doc.uri)
                    else doc.uri
                )
                nifti_image = self._extract_array(source_fn)
                if nifti_image is None:
                    continue
                doc.blob = nifti_image

    def _extract_array(self, uri):
        img = None
        try:
            img, meta = self.reader.get_data(self.reader.read(uri))
        except nibabel.filebasedimages.ImageFileError as e:
            file_extension = '.'.join(os.path.basename(uri).split('.')[1:])
            self.logger.error(f'Cannot work out file type of "*.{file_extension}"')
        except FileNotFoundError as e:
            self.logger.error(f'No such file or no access: "{uri}"')
        finally:
            return img

    def _save_uri_to_tmp_file(self, uri, tmpdir):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_fn = os.path.join(
            tmpdir,
            ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
            + '.nii.gz',
        )
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            with open(tmp_fn, 'wb') as f:
                f.write(binary_fn.read())
        return tmp_fn
