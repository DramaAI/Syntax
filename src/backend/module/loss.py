import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Self

class BinaryCrossEntropy(nn.Module):
    """
    BinaryCrossEntropy:
    A class to calculate the binary cross-entropy loss between predicted and target tensors.

    .. math::
        C = \sum{i=1} & number of classes 
        N ={} & batch size 

    """

    def __init__(self , head_type) -> None:
        """
        Initializes the BinaryCrossEntropy class.

        Parameters:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
        """
        super(BinaryCrossEntropy, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean") if head_type == "linear" else nn.BCELoss(reduction="mean")  # Assigns the binary cross-entropy function

    def forward(self , pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the binary cross-entropy loss.

        Parameters:
        pred (torch.Tensor): Predicted tensor
        target (torch.Tensor): Target tensor

        Returns:
        torch.Tensor: Binary cross-entropy loss
        """
        return self.bce(pred, target)  # Computes binary cross-entropy loss


class CrossEntropy(nn.Module):
    """
    CrossEntropy:
    A class to compute the categorical (multi-class) cross-entropy loss.
    """

    def __init__(self , head_type) -> None:
        """
        Initializes the CrossEntropy class.

        Parameters:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        """
        super(CrossEntropy, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean") if head_type == "linear" else F.cross_entropy(reduction="mean")  # Assigns the categorical cross-entropy function

    def forward(self , pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the categorical cross-entropy loss.

        Parameters:
        pred (torch.Tensor): Predicted tensor 
        target (torch.Tensor): Target tensor

        Returns:
        torch.Tensor: Categorical cross-entropy loss
        """
        return self.ce(pred, target)  # Computes categorical cross-entropy loss


class JointCrossEntropy(nn.Module):
    """
    JointCrossEntropy:
    A class that combines binary and categorical cross-entropy losses.
    """

    def __init__(self , head_type : str) -> None:
        """
        Initializes the JointCrossEntropy class.

        Parameters:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        """
        super().__init__()
        self.bce = BinaryCrossEntropy(head_type=head_type)  # Creates an instance of BinaryCrossEntropy
        self.ce = CrossEntropy(head_type=head_type)  # Creates an instance of CrossEntropy

    def forward(self              , 
                synonym           : torch.Tensor,  
                replacement       : torch.Tensor, 
                synonym_targets   : torch.Tensor, 
                replacement_target: torch.Tensor) -> torch.Tensor:
        """
        Combines binary and categorical cross-entropy losses and returns their sum.

        Parameters:
        replacement (torch.Tensor): Tensor of replacement probability [0, 1]
        synonym (torch.Tensor): Tensor for synonym of each s_i a probability distribution 
        synonym_target (torch.Tensor): Binary tensor, 1 denoting changed, and 0 not changed within labeled set
        replacement_target (torch.Tensor): Tensor for synonym of each y_i being a vector of vocab size of the one-hot encoded vector.

        Returns:
        torch.Tensor: Sum of binary and categorical cross-entropy losses
        """
        L_1 = self.bce(replacement, replacement_target)  # Calculates binary cross-entropy loss
        L_2 = self.ce(synonym, synonym_targets)  # Calculates categorical cross-entropy loss

        return L_1 + L_2  # Returns the sum of both losses
