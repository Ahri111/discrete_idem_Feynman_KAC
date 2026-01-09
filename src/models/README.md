# Crystal Graph Neural Network Models

Multi-framework implementation of Crystal Graph Convolutional Neural Networks with Graphormer attention.

## Supported Frameworks

| Framework | Files | Status |
|-----------|-------|--------|
| PyTorch | `components/symdiff/cgformer.py` | ✅ Full support |
| JAX/Flax | `model_jax.py`, `data_jax.py` | ✅ Full support |
| C++ (LibTorch) | `cpp_inference.py` | ✅ Inference only |
| C++ (ONNX) | `cpp_inference.py` | ✅ Inference only |

## Quick Start

### PyTorch (기본)

```python
from src.models import CGFormerEncoder, CheckpointManager

# 모델 생성
model = CGFormerEncoder(
    orig_atom_fea_len=92,
    nbr_fea_len=41,
    atom_fea_len=64,
    n_conv=3
)

# Pre-trained 모델 로드
manager = CheckpointManager("./checkpoints")
manager.load_model(model, "pretrained.pt")
```

### JAX/Flax

```python
from src.models import CrystalGraphConvNetJAX, CIFDataJAX
import jax

# 데이터 로드
dataset = CIFDataJAX(root_dir="./data", max_num_nbr=12, radius=8.0)
batch = dataset.get_batch([0, 1, 2])

# 모델 생성
model = CrystalGraphConvNetJAX(
    orig_atom_fea_len=92,
    nbr_fea_len=41,
    atom_fea_len=64,
    n_conv=3
)

# 초기화
rng = jax.random.PRNGKey(0)
params = model.init(rng,
    atom_fea=batch['atom_fea'],
    nbr_fea=batch['nbr_fea'],
    nbr_fea_idx=batch['nbr_fea_idx'],
    crystal_atom_idx=batch['crystal_atom_idx'],
    train=False
)
```

### C++ Export

```python
from src.models import export_for_cpp

# TorchScript + ONNX 내보내기
paths = export_for_cpp(
    model=model,
    output_dir="./exported",
    example_inputs={
        'atom_fea': atom_fea,
        'nbr_fea': nbr_fea,
        'nbr_fea_idx': nbr_fea_idx,
        'crystal_atom_idx': crystal_atom_idx
    }
)
# 생성됨: model_scripted.pt, model.onnx, cgnn_inference.hpp
```

## Pre-trained 모델 (.pt 파일)

### 지원하는 체크포인트 형식

```python
# 형식 1: state_dict만
torch.save(model.state_dict(), "model.pt")

# 형식 2: 전체 체크포인트 (권장)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, "checkpoint.pt")
```

### 체크포인트 로드

```python
from src.models import load_pretrained, CheckpointManager

# 간단한 로드
model = load_pretrained(model, "pretrained.pt", strict=False)

# 고급 사용
manager = CheckpointManager("./checkpoints")
checkpoint = manager.load("best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']}")
```

## 제한사항

### C++ 제한사항

| 기능 | Python | C++ |
|------|--------|-----|
| CIF 파싱 | ✅ pymatgen | ❌ 불가 |
| 전처리 | ✅ | ❌ Python 필요 |
| 모델 추론 | ✅ | ✅ |
| 학습 | ✅ | ⚠️ 제한적 |

**권장 워크플로우:**
1. Python에서 CIF 파일 전처리 → `.npz` 저장
2. C++에서 `.npz` 로드 → 추론 실행

### JAX 제한사항

- `pymatgen`은 Python 전용 (CIF 파싱은 Python에서만 가능)
- PyTorch → JAX 가중치 변환 시 수동 검증 필요
- BatchNorm 동작이 PyTorch와 약간 다를 수 있음

## 파일 구조

```
src/models/
├── __init__.py              # 통합 imports
├── README.md                # 이 문서
├── checkpoint_utils.py      # PyTorch 체크포인트 관리
├── data_jax.py              # JAX 데이터 로딩
├── model_jax.py             # JAX/Flax 모델
├── cpp_inference.py         # C++ 내보내기 유틸리티
└── components/
    └── symdiff/
        └── cgformer.py      # PyTorch 모델 (원본)
```

## 의존성

```bash
# PyTorch (필수)
pip install torch torch_geometric

# JAX (선택)
pip install jax jaxlib flax optax

# ONNX 내보내기 (선택)
pip install onnx onnxruntime
```

## API Reference

자세한 API 문서는 각 파일의 docstring을 참조하세요.
