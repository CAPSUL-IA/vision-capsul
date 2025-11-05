import os
import sys
import unittest
import torch
import pandas as pd
from unittest.mock import patch
from torchvision import transforms

sys.path.insert(0, '/app/src')

from process.data_loader import ImageClassificationDataset

TEST_IMAGES_PATH = '/tests/capsula/data/images/'

class TestImageClassificationDataset(unittest.TestCase):

    """
    Con @path le estamos indicando qué método queremos simular;
    en este caso la clase necesita un .csv por lo que cuando haga
    uso de la funcion de pandas read_csv esa función devolverá el
    DataFrame que hemos creado para este test en concreto.
    """
    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "IMAGE_NAME": ["test_1.jpg", "test_2.png", "test_3.jpg"],
            "LABELS": ["number", "apple", "banana"]
        })
        label_encoding = {"number": 0, "apple": 1, "banana": 2}
        self.dataset_multiclass = ImageClassificationDataset(csv_file="dummy.csv",
                                                             img_dir=TEST_IMAGES_PATH)
        self.dataset_multiclass.label_encoding = label_encoding
        

    def test_len(self):
        self.assertEqual(len(self.dataset_multiclass), 3)

    def test_getitem(self):
        image, _ = self.dataset_multiclass[1]
        self.assertNotIsInstance(image, torch.Tensor)

if __name__ == "__main__":
    unittest.main()
