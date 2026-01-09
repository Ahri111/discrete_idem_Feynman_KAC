"""
C++ Inference utilities for Crystal Graph Neural Networks.

This module provides:
1. TorchScript export for C++ LibTorch inference
2. ONNX export for cross-platform C++ inference
3. Data preprocessing utilities that can be called from C++

Limitations:
- CIF file parsing MUST remain in Python (pymatgen dependency)
- C++ can only handle pre-processed tensor data
- For full pipeline: Python preprocessing → C++ inference

Usage workflow:
1. Preprocess CIF files in Python to generate tensor data
2. Export model to TorchScript or ONNX
3. Load and run inference in C++

Requirements:
    pip install torch onnx onnxruntime
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn


class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript for C++ LibTorch.

    TorchScript models can be loaded directly in C++ using:
        torch::jit::load("model.pt")

    Example:
        exporter = TorchScriptExporter()
        exporter.export_model(model, "model_scripted.pt", example_inputs)
    """

    @staticmethod
    def export_model(
        model: nn.Module,
        save_path: str,
        example_inputs: Dict[str, torch.Tensor],
        method: str = 'trace',
        optimize: bool = True
    ) -> str:
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model
            save_path: Path to save the TorchScript model
            example_inputs: Dictionary of example inputs for tracing
            method: 'trace' or 'script'
            optimize: Whether to optimize the model

        Returns:
            Path to saved model
        """
        model.eval()

        if method == 'trace':
            # Trace-based export (works for most models)
            scripted_model = TorchScriptExporter._trace_model(model, example_inputs)
        elif method == 'script':
            # Script-based export (handles control flow)
            scripted_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        scripted_model.save(save_path)

        # Save metadata
        metadata_path = save_path.replace('.pt', '_metadata.json')
        TorchScriptExporter._save_metadata(model, example_inputs, metadata_path)

        return save_path

    @staticmethod
    def _trace_model(model: nn.Module, example_inputs: Dict[str, torch.Tensor]):
        """Trace model with example inputs."""
        # Create a wrapper to handle dict inputs
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx_flat, crystal_atom_idx_lengths):
                # Reconstruct crystal_atom_idx from flattened representation
                crystal_atom_idx = []
                offset = 0
                for length in crystal_atom_idx_lengths:
                    crystal_atom_idx.append(
                        crystal_atom_idx_flat[offset:offset + length]
                    )
                    offset += length

                return self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        wrapper = ModelWrapper(model)

        # Flatten crystal_atom_idx for tracing
        crystal_atom_idx = example_inputs['crystal_atom_idx']
        crystal_atom_idx_flat = torch.cat(crystal_atom_idx)
        crystal_atom_idx_lengths = torch.tensor([len(idx) for idx in crystal_atom_idx])

        traced = torch.jit.trace(
            wrapper,
            (
                example_inputs['atom_fea'],
                example_inputs['nbr_fea'],
                example_inputs['nbr_fea_idx'],
                crystal_atom_idx_flat,
                crystal_atom_idx_lengths
            )
        )

        return traced

    @staticmethod
    def _save_metadata(model: nn.Module, example_inputs: Dict[str, torch.Tensor], path: str):
        """Save model metadata for C++ loading."""
        metadata = {
            'input_shapes': {
                k: list(v.shape) if hasattr(v, 'shape') else [len(v)]
                for k, v in example_inputs.items()
            },
            'input_dtypes': {
                k: str(v.dtype) if hasattr(v, 'dtype') else 'list'
                for k, v in example_inputs.items()
            },
            'model_class': model.__class__.__name__,
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


class ONNXExporter:
    """
    Export PyTorch models to ONNX format for cross-platform inference.

    ONNX models can be loaded in C++ using:
    - ONNX Runtime C++ API
    - TensorRT (for NVIDIA GPUs)
    - OpenVINO (for Intel hardware)

    Example:
        exporter = ONNXExporter()
        exporter.export_model(model, "model.onnx", example_inputs)
    """

    @staticmethod
    def export_model(
        model: nn.Module,
        save_path: str,
        example_inputs: Dict[str, torch.Tensor],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model
            save_path: Path to save the ONNX model
            example_inputs: Dictionary of example inputs
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification for variable batch size

        Returns:
            Path to saved model
        """
        model.eval()

        # Create wrapper for simplified export
        class SimplifiedModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, atom_fea, nbr_fea, nbr_fea_idx, num_atoms_per_crystal):
                # Simplified forward that takes flat inputs
                # crystal_atom_idx reconstructed from num_atoms_per_crystal
                crystal_atom_idx = []
                offset = 0
                for num_atoms in num_atoms_per_crystal:
                    crystal_atom_idx.append(
                        torch.arange(offset, offset + num_atoms, device=atom_fea.device)
                    )
                    offset += num_atoms

                return self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        wrapped_model = SimplifiedModel(model)

        # Prepare inputs for export
        crystal_atom_idx = example_inputs['crystal_atom_idx']
        num_atoms_per_crystal = torch.tensor([len(idx) for idx in crystal_atom_idx])

        input_names = ['atom_fea', 'nbr_fea', 'nbr_fea_idx', 'num_atoms_per_crystal']
        output_names = ['prediction']

        if dynamic_axes is None:
            dynamic_axes = {
                'atom_fea': {0: 'num_atoms'},
                'nbr_fea': {0: 'num_atoms'},
                'nbr_fea_idx': {0: 'num_atoms'},
                'num_atoms_per_crystal': {0: 'batch_size'},
                'prediction': {0: 'batch_size'}
            }

        torch.onnx.export(
            wrapped_model,
            (
                example_inputs['atom_fea'],
                example_inputs['nbr_fea'],
                example_inputs['nbr_fea_idx'],
                num_atoms_per_crystal
            ),
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )

        # Verify the model
        ONNXExporter._verify_onnx(save_path)

        return save_path

    @staticmethod
    def _verify_onnx(model_path: str):
        """Verify ONNX model validity."""
        try:
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            print(f"ONNX model verified: {model_path}")
        except ImportError:
            print("Warning: onnx package not installed. Skipping verification.")
        except Exception as e:
            print(f"Warning: ONNX verification failed: {e}")


class DataPreprocessor:
    """
    Preprocess crystal data for C++ inference.

    This class generates binary files that can be loaded efficiently in C++.
    The preprocessing includes:
    - Atom feature extraction
    - Neighbor list construction
    - Gaussian distance expansion

    Output format: NumPy .npz files (can be loaded in C++ with cnpy library)
    """

    def __init__(
        self,
        atom_init_file: str,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2
    ):
        """
        Initialize preprocessor.

        Args:
            atom_init_file: Path to atom_init.json
            max_num_nbr: Maximum number of neighbors
            radius: Cutoff radius for neighbor search
            dmin: Minimum distance for Gaussian expansion
            step: Step size for Gaussian expansion
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step

        # Load atom embeddings
        with open(atom_init_file) as f:
            self.atom_embeddings = json.load(f)
        self.atom_embeddings = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in self.atom_embeddings.items()
        }

        # Gaussian filter
        self.gaussian_filter = np.arange(dmin, radius + step, step)
        self.var = step

    def preprocess_structure(
        self,
        cif_path: str,
        target: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess a single crystal structure.

        Args:
            cif_path: Path to CIF file
            target: Target property value (optional)

        Returns:
            Dictionary with preprocessed numpy arrays
        """
        from pymatgen.core.structure import Structure

        crystal = Structure.from_file(cif_path)

        # Atom features
        atom_fea = np.vstack([
            self.atom_embeddings[crystal[i].specie.number]
            for i in range(len(crystal))
        ]).astype(np.float32)

        # Neighbor information
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx = []
        nbr_distances = []

        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_distances.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_distances.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int32)
        nbr_distances = np.array(nbr_distances, dtype=np.float32)

        # Gaussian distance expansion
        nbr_fea = np.exp(
            -(nbr_distances[..., np.newaxis] - self.gaussian_filter) ** 2 / self.var ** 2
        ).astype(np.float32)

        result = {
            'atom_fea': atom_fea,
            'nbr_fea': nbr_fea,
            'nbr_fea_idx': nbr_fea_idx,
            'num_atoms': np.array([len(crystal)], dtype=np.int32)
        }

        if target is not None:
            result['target'] = np.array([target], dtype=np.float32)

        return result

    def save_preprocessed(
        self,
        data: Dict[str, np.ndarray],
        save_path: str,
        format: str = 'npz'
    ):
        """
        Save preprocessed data to file.

        Args:
            data: Preprocessed data dictionary
            save_path: Output path
            format: 'npz' or 'binary'
        """
        if format == 'npz':
            np.savez(save_path, **data)
        elif format == 'binary':
            # Save as separate binary files for C++ loading
            base_path = Path(save_path)
            base_path.mkdir(parents=True, exist_ok=True)
            for key, arr in data.items():
                arr.tofile(base_path / f"{key}.bin")
                # Save shape info
                with open(base_path / f"{key}_shape.txt", 'w') as f:
                    f.write(' '.join(map(str, arr.shape)))
        else:
            raise ValueError(f"Unknown format: {format}")

    def batch_preprocess(
        self,
        cif_paths: List[str],
        targets: Optional[List[float]] = None,
        output_dir: str = "./preprocessed"
    ) -> str:
        """
        Preprocess multiple structures and save to directory.

        Args:
            cif_paths: List of CIF file paths
            targets: List of target values (optional)
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        all_atom_fea = []
        all_nbr_fea = []
        all_nbr_fea_idx = []
        all_num_atoms = []
        all_targets = []

        base_idx = 0

        for i, cif_path in enumerate(cif_paths):
            target = targets[i] if targets else None
            data = self.preprocess_structure(cif_path, target)

            all_atom_fea.append(data['atom_fea'])
            all_nbr_fea.append(data['nbr_fea'])
            all_nbr_fea_idx.append(data['nbr_fea_idx'] + base_idx)
            all_num_atoms.append(data['num_atoms'])

            if target is not None:
                all_targets.append(data['target'])

            base_idx += data['num_atoms'][0]

        # Concatenate all data
        batch_data = {
            'atom_fea': np.concatenate(all_atom_fea, axis=0),
            'nbr_fea': np.concatenate(all_nbr_fea, axis=0),
            'nbr_fea_idx': np.concatenate(all_nbr_fea_idx, axis=0),
            'num_atoms': np.concatenate(all_num_atoms, axis=0)
        }

        if all_targets:
            batch_data['targets'] = np.concatenate(all_targets, axis=0)

        # Save
        np.savez(os.path.join(output_dir, 'batch_data.npz'), **batch_data)

        return output_dir


# C++ Header Template
CPP_HEADER_TEMPLATE = '''
/*
 * Crystal Graph Neural Network Inference Header
 * Auto-generated by cpp_inference.py
 *
 * Usage:
 *   1. Install LibTorch or ONNX Runtime
 *   2. Include this header
 *   3. Load preprocessed data (NumPy .npz files using cnpy)
 *   4. Run inference
 *
 * Dependencies:
 *   - LibTorch (for TorchScript models)
 *   - ONNX Runtime (for ONNX models)
 *   - cnpy (for loading NumPy files)
 */

#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>

namespace cgnn {

class CrystalGraphInference {
public:
    explicit CrystalGraphInference(const std::string& model_path) {
        try {
            model_ = torch::jit::load(model_path);
            model_.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Failed to load model: " + std::string(e.what()));
        }
    }

    torch::Tensor predict(
        const torch::Tensor& atom_fea,        // (N, atom_fea_len)
        const torch::Tensor& nbr_fea,         // (N, M, nbr_fea_len)
        const torch::Tensor& nbr_fea_idx,     // (N, M)
        const torch::Tensor& crystal_atom_idx_flat,  // Flattened indices
        const torch::Tensor& crystal_atom_idx_lengths  // Lengths per crystal
    ) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(atom_fea);
        inputs.push_back(nbr_fea);
        inputs.push_back(nbr_fea_idx);
        inputs.push_back(crystal_atom_idx_flat);
        inputs.push_back(crystal_atom_idx_lengths);

        return model_.forward(inputs).toTensor();
    }

    void to(torch::Device device) {
        model_.to(device);
    }

private:
    torch::jit::script::Module model_;
};

// Utility function to load NumPy array (requires cnpy)
// torch::Tensor load_numpy(const std::string& path);

}  // namespace cgnn

/*
 * Example usage:
 *
 * int main() {
 *     cgnn::CrystalGraphInference model("model_scripted.pt");
 *
 *     // Load preprocessed data
 *     auto atom_fea = torch::randn({100, 92});
 *     auto nbr_fea = torch::randn({100, 12, 41});
 *     auto nbr_fea_idx = torch::randint(0, 100, {100, 12});
 *     auto crystal_idx_flat = torch::arange(100);
 *     auto crystal_idx_lengths = torch::tensor({50, 50});
 *
 *     auto prediction = model.predict(
 *         atom_fea, nbr_fea, nbr_fea_idx,
 *         crystal_idx_flat, crystal_idx_lengths
 *     );
 *
 *     std::cout << prediction << std::endl;
 *     return 0;
 * }
 */
'''


def generate_cpp_header(output_path: str = "cgnn_inference.hpp"):
    """Generate C++ header file for inference."""
    with open(output_path, 'w') as f:
        f.write(CPP_HEADER_TEMPLATE)
    print(f"Generated C++ header: {output_path}")
    return output_path


# Convenience function for complete export pipeline
def export_for_cpp(
    model: nn.Module,
    output_dir: str,
    example_inputs: Dict[str, torch.Tensor],
    formats: List[str] = ['torchscript', 'onnx']
) -> Dict[str, str]:
    """
    Export model in all formats for C++ inference.

    Args:
        model: PyTorch model
        output_dir: Output directory
        example_inputs: Example inputs for tracing
        formats: List of export formats

    Returns:
        Dictionary of format -> file path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    if 'torchscript' in formats:
        ts_path = os.path.join(output_dir, 'model_scripted.pt')
        TorchScriptExporter.export_model(model, ts_path, example_inputs)
        paths['torchscript'] = ts_path

    if 'onnx' in formats:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        ONNXExporter.export_model(model, onnx_path, example_inputs)
        paths['onnx'] = onnx_path

    # Generate header
    header_path = os.path.join(output_dir, 'cgnn_inference.hpp')
    generate_cpp_header(header_path)
    paths['header'] = header_path

    return paths
