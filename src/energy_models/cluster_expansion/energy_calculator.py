import pickle
import numpy as np
import module_octa_CE as cediff


class EnergyCalculator:
    """
    Cluster-Expansion based Energy Calculator
    """
    
    def __init__(self, model_file, scaler_file, cluster_file):
        
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
            
        