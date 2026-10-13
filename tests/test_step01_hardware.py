import importlib.metadata
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rank_refinement.runtime.hardware import inspect_hardware


class HardwareBranchTests(unittest.TestCase):
    def test_no_torch_auto_still_prepares_data(self):
        with patch('rank_refinement.runtime.hardware.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError):
            report=inspect_hardware('auto')
            self.assertEqual(report['future_model_device'],'cpu')
            self.assertEqual(report['preparation_device'],'cpu')

    def test_explicit_cuda_without_cuda_rejected(self):
        with patch('rank_refinement.runtime.hardware.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaises(RuntimeError):inspect_hardware('cuda')

    def test_auto_selects_available_gpu_but_preparation_is_cpu(self):
        fake=SimpleNamespace(version=SimpleNamespace(cuda='12.8'),cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,
            get_device_properties=lambda i:SimpleNamespace(name='synthetic_gpu_for_branch_test',total_memory=6*1024**3)))
        with patch('rank_refinement.runtime.hardware.importlib.metadata.version',return_value='2.7.0+cu128'),patch.dict(sys.modules,{'torch':fake}):
            report=inspect_hardware('auto')
            self.assertEqual(report['future_model_device'],'cuda')
            self.assertEqual(report['preparation_device'],'cpu')
            self.assertEqual(report['gpus'][0]['memory_gib'],6)

    def test_cpu_request_respected_with_gpu_available(self):
        fake=SimpleNamespace(version=SimpleNamespace(cuda='12.8'),cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:0))
        with patch('rank_refinement.runtime.hardware.importlib.metadata.version',return_value='2.7.0+cu128'),patch.dict(sys.modules,{'torch':fake}):
            self.assertEqual(inspect_hardware('cpu')['future_model_device'],'cpu')
