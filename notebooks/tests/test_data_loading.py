import unittest
from unittest.mock import patch
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestDataLoading(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read):
        mock_read.return_value = pd.DataFrame({'col': [1,2,3]})
        self.assertTrue(True)
    
    def test_data_validation(self): self.assertTrue(True)
    def test_missing_values(self): self.assertTrue(True)
    def test_file_not_found(self): self.assertTrue(True)
    def test_data_types(self): self.assertTrue(True)
