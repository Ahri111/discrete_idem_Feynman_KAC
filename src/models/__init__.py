"""
Models module for Crystal Graph Neural Networks.

Provides:
- PyTorch models (cgformer.py)
- JAX/Flax models (model_jax.py)
- Data loading utilities (data_jax.py)
- Checkpoint management (checkpoint_utils.py)
- C++ export utilities (cpp_inference.py)
"""

# PyTorch models
from .components.symdiff.cgformer import (
    ConvLayer,
    CentralityEncoding,
    GraphormerAttentionHead,
    GraphormerMultiHeadAttention,
    GraphormerEncoderLayer,
    GraphormerEncoder,
    CGFormerEncoder,
)

# Checkpoint utilities
from .checkpoint_utils import (
    CheckpointManager,
    load_pretrained,
    convert_checkpoint_format,
)

# C++ export utilities
from .cpp_inference import (
    TorchScriptExporter,
    ONNXExporter,
    DataPreprocessor,
    export_for_cpp,
    generate_cpp_header,
)

__all__ = [
    # PyTorch models
    'ConvLayer',
    'CentralityEncoding',
    'GraphormerAttentionHead',
    'GraphormerMultiHeadAttention',
    'GraphormerEncoderLayer',
    'GraphormerEncoder',
    'CGFormerEncoder',
    # Checkpoint utilities
    'CheckpointManager',
    'load_pretrained',
    'convert_checkpoint_format',
    # C++ export
    'TorchScriptExporter',
    'ONNXExporter',
    'DataPreprocessor',
    'export_for_cpp',
    'generate_cpp_header',
]

# JAX imports are optional (may not have JAX installed)
try:
    from .model_jax import (
        CrystalGraphConvNet as CrystalGraphConvNetJAX,
        CGFormerEncoder as CGFormerEncoderJAX,
        create_train_state,
        load_pytorch_weights,
    )
    from .data_jax import (
        CIFDataJAX,
        JAXDataLoader,
        get_train_val_test_loader_jax,
        torch_to_jax,
        jax_to_torch,
    )
    __all__.extend([
        'CrystalGraphConvNetJAX',
        'CGFormerEncoderJAX',
        'create_train_state',
        'load_pytorch_weights',
        'CIFDataJAX',
        'JAXDataLoader',
        'get_train_val_test_loader_jax',
        'torch_to_jax',
        'jax_to_torch',
    ])
except ImportError:
    pass  # JAX not installed
