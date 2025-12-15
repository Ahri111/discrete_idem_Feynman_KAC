from abc import ABC, abstractmethod
import torch

class EnergyOracle(ABC):
    
    @abstractmethod
    def compute_energy(self, x):
        pass
    
    @abstractmethod
    def to(self, device):
        pass
    
class ClusterExpansionOracle(EnergyOracle):
    """
    Lasso-based Cluster Expansion
    """
    def __