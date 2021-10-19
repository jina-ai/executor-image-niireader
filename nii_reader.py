from jina import Executor, DocumentArray, requests
from monai.data.image_reader import NibabelReader


class NiiReader(Executor):
    def __init__(self, *args, **kwargs):
        super(NiiReader, self).__init__()
        self.r = NibabelReader()

    @requests
    def read(self, docs, **kwargs):
        for doc in docs:
            doc.blob = self._extract_array(doc.uri)

    def _extract_array(self, uri):
        img, _ = self.r.get_data(self.r.read(uri))
        return img
