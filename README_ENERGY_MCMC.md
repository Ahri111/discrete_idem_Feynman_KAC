# Energy Evaluation & MCMC Sampling

배치 처리를 지원하는 에너지 평가 및 MCMC 샘플링 구현

## 주요 기능

### 1. EnergyCalculator (배치 지원)
- **단일 구조 에너지 계산**: `compute_energy(poscar_file)`
- **배치 에너지 계산**: `compute_energy_batch(poscar_files, batch_size=32)`
- **피처 기반 계산**: `compute_energy_from_features(features)`

### 2. ClusterExpansionOracle
- PyTorch 텐서 지원
- 배치 처리 지원
- GPU/CPU 디바이스 관리

### 3. MCMCSampler
- Metropolis-Hastings 알고리즘
- 단일 및 배치 MCMC 샘플링
- 에너지 히스토리 추적
- 수락률(acceptance rate) 통계

## 사용 예제

### 기본 에너지 계산

```python
from src.energy_models.cluster_expansion.energy_calculator import EnergyCalculator

# 초기화
calculator = EnergyCalculator(
    model_file='path/to/trained_lasso_model.pkl',
    scaler_file='path/to/trained_lasso_scaler.pkl',
    cluster_file='path/to/reference_clusters.json'
)

# 단일 구조
energy = calculator.compute_energy('POSCAR')
print(f"Energy: {energy} eV")

# 배치 처리 (여러 구조를 효율적으로 계산)
poscar_files = ['POSCAR1', 'POSCAR2', 'POSCAR3', ...]
energies = calculator.compute_energy_batch(poscar_files, batch_size=32)
```

### Oracle 사용 (PyTorch 통합)

```python
from src.energy_models.oracles import ClusterExpansionOracle

# 초기화
oracle = ClusterExpansionOracle(
    model_file='path/to/model.pkl',
    scaler_file='path/to/scaler.pkl',
    cluster_file='path/to/clusters.json',
    device='cuda'  # or 'cpu'
)

# 단일 구조
energy_tensor = oracle.compute_energy('POSCAR')

# 배치
energies = oracle.compute_energy_batch(poscar_files, batch_size=32)

# 피처 텐서로부터 직접 계산
features = torch.randn(batch_size, n_features)
energies = oracle(features)
```

### MCMC 샘플링

```python
from src.energy_models.cluster_expansion.mcmc_sampler import MCMCSampler

# 초기화
sampler = MCMCSampler(
    energy_calculator=calculator,
    temperature=1000.0,  # Kelvin
    swap_types=[(0, 2)],  # A-site atom swaps
    random_seed=42
)

# 단일 구조 MCMC
trajectory = sampler.run(
    initial_poscar_file='POSCAR',
    n_steps=10000,
    output_dir='mcmc_results',
    save_interval=100,
    verbose=True
)

# 통계 확인
print(f"Acceptance rate: {sampler.get_acceptance_rate():.3f}")
print(f"Energy history: {sampler.energy_history}")
```

### 배치 MCMC 샘플링

```python
# 여러 초기 구조에 대해 MCMC 실행
trajectories = sampler.run_batch(
    initial_poscar_files=['POSCAR1', 'POSCAR2', 'POSCAR3'],
    n_steps=5000,
    output_dirs=['mcmc_1', 'mcmc_2', 'mcmc_3'],
    save_interval=100,
    verbose=True
)

# 각 궤적 분석
for i, traj in enumerate(trajectories):
    final_energy = traj[-1][0]
    print(f"Chain {i+1} final energy: {final_energy:.6f} eV")
```

## 배치 처리의 장점

1. **효율성**: 여러 구조를 한번에 처리하여 오버헤드 감소
2. **병렬화**: 피처 스케일링 및 예측을 벡터화된 연산으로 수행
3. **메모리 관리**: 배치 크기를 조절하여 메모리 사용량 제어
4. **실패 처리**: 일부 구조가 실패해도 나머지는 계속 처리

## 파일 구조

```
src/energy_models/
├── oracles.py                           # EnergyOracle, ClusterExpansionOracle
└── cluster_expansion/
    ├── energy_calculator.py             # EnergyCalculator (배치 지원)
    ├── mcmc_sampler.py                  # MCMCSampler (배치 지원)
    ├── cluster_counter.py               # 클러스터 카운팅
    ├── structure_utils.py               # POSCAR 읽기/쓰기
    └── energy_parameter/
        ├── trained_lasso_model.pkl      # 학습된 모델
        ├── trained_lasso_scaler.pkl     # 스케일러
        └── reference_clusters.json      # 참조 클러스터
```

## 예제 실행

```bash
# 예제 스크립트 실행
python examples/energy_mcmc_example.py
```

## API 요약

### EnergyCalculator

| 메서드 | 설명 | 입력 | 출력 |
|--------|------|------|------|
| `compute_energy` | 단일 구조 에너지 계산 | `poscar_file: str` | `float` |
| `compute_energy_batch` | 배치 에너지 계산 | `poscar_files: List[str]`, `batch_size: int` | `List[float]` |
| `compute_energy_from_features` | 피처로부터 계산 | `features: ndarray` | `float or ndarray` |

### ClusterExpansionOracle

| 메서드 | 설명 | 입력 | 출력 |
|--------|------|------|------|
| `compute_energy` | 다양한 입력 타입 지원 | `x: str/List/Tensor/ndarray` | `Tensor` |
| `compute_energy_single` | 단일 구조 | `poscar_file: str` | `Tensor` |
| `compute_energy_batch` | 배치 처리 | `poscar_files: List[str]` | `Tensor` |
| `to` | 디바이스 이동 | `device: str` | `self` |

### MCMCSampler

| 메서드 | 설명 | 입력 | 출력 |
|--------|------|------|------|
| `run` | 단일 MCMC 실행 | `initial_poscar_file`, `n_steps`, ... | `trajectory: List` |
| `run_batch` | 배치 MCMC 실행 | `initial_poscar_files`, `n_steps`, ... | `List[trajectory]` |
| `get_acceptance_rate` | 수락률 조회 | - | `float` |

## 성능 최적화 팁

1. **배치 크기 조정**: 메모리와 속도의 균형을 맞추세요 (권장: 16-64)
2. **거리 행렬 재사용**: MCMC에서 `dismatswap` 사용으로 효율성 향상
3. **GPU 사용**: 큰 배치의 경우 `device='cuda'` 설정
4. **저장 간격**: `save_interval`을 크게 설정하여 I/O 감소

## 의존성

- numpy
- torch
- pickle
- sklearn (StandardScaler, Lasso model)

## 문의사항

구현 관련 질문이나 버그 리포트는 이슈로 등록해주세요.
