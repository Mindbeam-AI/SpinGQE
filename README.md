# SpinGQE - A Generative Quantum Eigensolver for Spin Hamiltonians

[![arXiv](https://img.shields.io/badge/arXiv-2603.24298-b31b1b.svg)](https://arxiv.org/abs/2603.24298)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"SpinGQE - A generative quantum eigensolver for spin Hamiltonians"** by Alexander Holden, Moinul Hossain Rahat, and Nii Osae Osae Dade.

## Overview

This repository implements the SpinGQE, a novel approach to quantum ground state search that reframes circuit design as a generative modeling task. Instead of optimizing continuous parameters within a fixed ansatz like traditional VQE methods, SpinGQE uses a transformer-based decoder to learn distributions over quantum circuits that produce low-energy states for spin Hamiltonians.

**Key Features:**
- **Generative approach**: Transformer model learns to generate quantum circuits
- **Energy-guided training**: Weighted MSE loss between model logits and circuit energies
- **Post-processing optimization**: Hybrid classical-quantum refinement for precision
- **Validated on Heisenberg model**: Achieves near-exact ground states on 4-qubit systems

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Mindbeam-AI/SpinGQE.git
cd SpinGQE
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the VQE ground state dataset:
```bash
git clone https://github.com/Qulacs-Osaka/VQE-generated-dataset.git
```

Ensure the dataset is placed in your working directory with the structure:
```
./VQE-generated-dataset/data/ground_state/
```

## Quick Start

### Training a Model

Run the training script with default parameters for the Heisenberg model:

```bash
python train.py
```

This will:
- Train a 4-layer transformer with 8 attention heads
- Use the 4-qubit Heisenberg Hamiltonian (label 1)
- Generate circuits of length 12
- Train for 700 epochs
- Save results to `heisenberg/`

### Key Hyperparameters

Edit the following parameters in `train.py` to customize your experiment:

```python
# System configuration
num_qubits = 4          # Number of qubits
ham_label = 1           # Hamiltonian type (0-5)
seq_len = 12            # Circuit length (number of gates)

# Training parameters
n_epochs = 700          # Training epochs
seq_gen = 10            # Circuits generated per epoch
n_batches = 10          # Batches per epoch
temperature = 0.5       # Generation temperature
beta = 0.3              # Energy weighting parameter

# Model architecture
n_layer = 12            # Transformer layers
n_head = 8              # Attention heads
n_embd = 512            # Embedding dimension
```

## Repository Structure

```
SpinGQE/
├── train.py                # Main training script
├── SpinGQE.py              # GQE model class with loss functions
├── model.py                # Transformer architecture (nano-GPT based)
├── hamiltonian.py          # Hamiltonian generation utilities
├── requirements.txt        # Python dependencies
├── postprocessing.ipynb    # Postprocessing optimization algorithms
└── README.md               # This file
```

### File Descriptions

- **`train.py`**: Main entry point for training. Handles data generation, training loop, evaluation, and checkpointing.
- **`SpinGQE.py`**: Extends the base GPT model with GQE-specific functionality:
  - Weighted MSE loss for energy matching
  - Alternative loss functions (GRPO, DPO, CPO)
  - Circuit generation with logit accumulation
- **`model.py`**: Transformer architecture based on nano-GPT
- **`hamiltonian.py`**: Functions to generate various spin Hamiltonians (labels 0-5)
- **`postprocessing.ipynb`**: Contains final model evaluation, checkpoint averaging and postprocessing optimization algorithms


## Usage Examples

### 1. Training on Different Hamiltonians

The codebase supports 6 predefined Hamiltonian types (labels 0-5):

```python
# Label 0: ZZ + X interactions
ham_label = 0

# Label 1: Heisenberg model (XX + YY + ZZ + Z)
ham_label = 1

# Label 2: Alternating coupling Heisenberg
ham_label = 2

# Label 3: Long-range Heisenberg
ham_label = 3

# Label 4: Fermionic Hubbard model (even qubits only)
ham_label = 4

# Label 5: 2D Hubbard ladder (4N qubits only)
ham_label = 5
```

### 2. Customizing the Operator Pool

Modify `build_operator_pool()` in `train.py` to change available gates:

```python
def build_operator_pool(n_qubits, t_values=None):
    if t_values is None:
        # Customize rotation angles
        t_values = [np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32]
        t_values += [-t for t in t_values]
    
    pool = []
    
    # Two-qubit gates (nearest-neighbor)
    for i in range(n_qubits - 1):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'ZZ', wires=[i, i + 1]))
            pool.append(qml.PauliRot(t, 'XX', wires=[i, i + 1]))
            pool.append(qml.PauliRot(t, 'YY', wires=[i, i + 1]))
    
    # Single-qubit gates
    for i in range(n_qubits):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'Z', wires=i))
    
    return pool
```

### 3. Loading and Continuing Training

Resume training from a checkpoint:

```python
load_model = True
load_dir = "heisenberg_betas_test/0.1"

# Model and optimizer will be loaded automatically
```

### 4. Evaluation and Visualization

The training script automatically generates:
- **Loss plots** (`loss_fig.png`)
- **Energy convergence plots** (`eval_fig.png`, `gen_fig.png`)
- **Energy histograms** (saved to `{dir}/histo/`)
- **Training data** (CSV files with energies and losses)

## Reproducing Paper Results

### Antiferromagnetic Regime (J = h = 10)

```python
# In train.py, set:
ham_label = 1
num_qubits = 4
beta = 0.3
seq_gen = 80  # M = 10 circuits per epoch (80/8 batches)
n_batches = 10
seq_len = 12
temperature = 0.5
n_epochs = 700

# Model: 12 layers, 8 heads (37M parameters)
n_layer = 12
n_head = 8
n_embd = 512

ham = gen_hamiltonian(ham_label, num_qubits, 10, 10)
```

Expected results:
- Pre-optimization: ~-60.78 J
- Post-optimization: ~-64.64 J (near-exact)

### Field-Dominated Regime (J = 1, h = 10)

Modify the Hamiltonian parameters in the gen_hamiltonian function:

```python
ham = gen_hamiltonian(ham_label, num_qubits, 1, 10)
```

Expected results:
- Converges to -37.0 J without post-processing

## Model Architecture

The SpinGQE framework consists of:

1. **Transformer Decoder**: Generates sequences of quantum gates
   - Input: Token representing gate from operator pool
   - Output: Distribution over next gate

2. **Energy Evaluation**: Quantum circuit execution
   - Evaluates energy at each gate subsequence
   - Provides training signal

3. **Weighted Loss Function**:
   ```
   w(E, β) = 1 / (1 + e^(βE))
   MSE_weighted = (1/N) Σ w(E_N) Σ (l_t - E_t)²
   ```

4. **Post-Processing** (optional):
   - Angle refinement (L-BFGS-B)
   - Greedy wire reassignment

## Configuration

### Model Sizes

| Configuration | Layers | Heads | Embedding | Parameters |
|--------------|--------|-------|-----------|------------|
| Small        | 4      | 8     | 384       | ~7M        |
| Medium       | 12     | 8     | 512       | ~37M       |
| Large        | 16     | 16    | 768       | ~113M      |

**Recommendation**: Use Medium (37M) for best balance of performance and training stability.

### Energy Weighting (β)

- **Low β (0.1-0.3)**: Broader exploration, prevents local minima
- **High β (1.0-2.0)**: Focused on low-energy regions
- **Optimal**: β = 0.3 for Heisenberg model

### Generation Parameters

- **Temperature**: Controls sampling diversity
  - Lower (0.1-0.5): Exploitation of learned distribution
  - Higher (0.8-1.0): Exploration of circuit space
- **Sequence Length**: Number of gates per circuit (8-16 typical)

## Output Files

Training produces the following outputs in the specified directory:

```
{dir}/
├── final.pt              # Final model checkpoint
├── config.json           # Model configuration
├── metadata.json         # Training hyperparameters and results
├── losses.csv            # Training loss history
├── pred_Es_t.csv         # Predicted energies during evaluation
├── true_Es_t.csv         # Measured energies during evaluation
├── loss_fig.png          # Loss convergence plot
├── eval_fig.png          # Energy evaluation plot
├── gen_fig.png           # Generated training data plot
└── histo/                # Energy distribution histograms
    ├── 50.png
    ├── 100.png
    └── ...
```

## Advanced Usage

### Multi-GPU Training

The code supports CUDA by default. For multi-GPU:

```python
model = torch.nn.DataParallel(model)
```

### Custom Hamiltonians

Add your own Hamiltonian to `hamiltonian.py`:

```python
def custom_hamiltonian(num_qubits):
    ops = []
    coeffs = []
    
    # Define your Hamiltonian terms
    # Example: Custom coupling pattern
    for i in range(num_qubits-1):
        ops.append(PauliZ(i) @ PauliZ(i+1))
        coeffs.append(your_coupling_strength)
    
    return Hamiltonian(coeffs, ops)
```

Then update `gen_hamiltonian()` to include your label.

## Performance Tips

1. **Start with smaller models**: 7M parameter model is often sufficient
2. **Use moderate β values**: 0.3 works well across regimes
3. **Keep sequence length manageable**: 12 gates is a good default
4. **Monitor energy histograms**: Check for convergence and diversity
5. **Save checkpoints frequently**: Every 50 epochs is recommended

## Troubleshooting

### Common Issues

**Problem**: Model doesn't converge to low energies

**Solutions**:
- Reduce β (try 0.1-0.3)
- Increase number of circuits per epoch
- Check operator pool contains relevant gates
- Verify Hamiltonian is correctly defined

**Problem**: Training is unstable

**Solutions**:
- Use smaller model (7M or 37M parameters)
- Reduce learning rate
- Increase batch size
- Add gradient clipping

**Problem**: Out of memory errors

**Solutions**:
- Reduce batch size (`n_batches`)
- Reduce model size
- Use gradient checkpointing
- Train on CPU (slower)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{holden2026gqe,
  title={SpinGQE - A generative quantum eigensolver for spin Hamiltonians},
  author={Holden, Alexander and Rahat, Moinul Hossain and Dade, Nii Osae Osae},
  journal={arXiv preprint arXiv:2603.24298},
  year={2026}
}
```

## Related Work

This implementation builds upon:
- **PennyLane GQE Tutorial**: [https://pennylane.ai/qml/demos/gqe_training](https://pennylane.ai/qml/demos/gqe_training)
- **VQE Ground State Dataset**: [https://github.com/Qulacs-Osaka/VQE-generated-dataset](https://github.com/Qulacs-Osaka/VQE-generated-dataset)
- **nano-GPT**: Transformer architecture base

## Contact

For questions or issues:
- **Email**: research@mindbeam.ai
- **Issues**: [GitHub Issues](https://github.com/Mindbeam-AI/SpinGQE/issues)

## Acknowledgments

We thank the authors of the VQE ground state dataset for making their data publicly available.

---

**Mindbeam AI** | [Website](https://mindbeam.ai)
