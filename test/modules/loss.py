import unittest
import pathlib
import sys
import os

root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(os.path.join(root))

from src.backend.modules.loss import BinaryCrossEntropy, CrossEntropy

import torch
import torch.nn.functional as F

class TestLossModules(unittest.TestCase):
    def test_binary_cross_entropy(self):
        bce = BinaryCrossEntropy()
        pred = torch.tensor([0.8, 0.4, 0.2])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = bce(pred, target)
        expected_loss = F.binary_cross_entropy(pred, target, reduction='mean')

        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_categorical_cross_entropy(self):
        ce = CrossEntropy()
        pred = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]])
        target = torch.tensor([0, 2])
        loss = ce(pred, target)
        expected_loss = F.cross_entropy(pred, target, reduction="mean")
        self.assertTrue(torch.allclose(loss, expected_loss))

