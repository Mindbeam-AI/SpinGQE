from pennylane import Hamiltonian, PauliX, PauliY, PauliZ, FermiA, FermiC, jordan_wigner, to_openfermion
from qiskit import QuantumCircuit

def sum_to_hamiltonian(op_sum):
    terms = op_sum.terms()  # list of ([coeff], [observable])
    coeffs = [c for c in terms[0]]
    ops = [o for o in terms[1]]
    return Hamiltonian(coeffs, ops)


def zero(num_qubits, j1=1, j2=2):
    ops = []
    coeffs = []
    for i in range(num_qubits-1):
        ops.append(PauliZ(i) @ PauliZ(i+1))
        coeffs.append(j1)
    for i in range(num_qubits):
        ops.append(PauliX(i))
        coeffs.append(j2)
    return Hamiltonian(coeffs, ops)


def one(num_qubits, j1=1, j2=2):
  ops = []
  coeffs = []
  for i in range(num_qubits-1):
    ops.append(PauliX(i) @ PauliX(i+1) + PauliY(i) @ PauliY(i+1) + PauliZ(i) @ PauliZ(i+1))
    coeffs.append(j1)
  for i in range(num_qubits):
    ops.append(PauliZ(i))
    coeffs.append(j2)
  return Hamiltonian(coeffs, ops)

def two(num_qubits):
  ops = []
  coeffs = []
  for i in range(num_qubits-1):
    ops.append(PauliX(i) @ PauliX(i+1) + PauliY(i) @ PauliY(i+1) + PauliZ(i) @ PauliZ(i+1))
    coeffs.append(1+1.5*pow(-1, i))
  return Hamiltonian(coeffs, ops)

def three(num_qubits, j1=1, j2=3):
  ops = []
  coeffs = []
  for i in range(num_qubits-1):
    ops.append(PauliX(i) @ PauliX(i+1) + PauliY(i) @ PauliY(i+1) + PauliZ(i) @ PauliZ(i+1))
    coeffs.append(j1)
  for i in range(num_qubits-2):
    ops.append(PauliX(i) @ PauliX(i+2) + PauliY(i) @ PauliY(i+2) + PauliZ(i) @ PauliZ(i+2))
    coeffs.append(j2)
  return Hamiltonian(coeffs, ops)

def four(num_qubits):
  if num_qubits % 2 == 1:
    return None

  num_sites = num_qubits // 2
  H_f = 0.0

  for j in range(num_sites - 1):
    for spin in [0, 1]:
      i = 2 * j + spin
      k = 2 * (j + 1) + spin

      hop = FermiC(i) * FermiA(k)
      hop_hc = FermiC(k) * FermiA(i)
      H_f -= (hop + hop_hc)

  for z in range(num_sites):
    i_up = 2 * z
    i_down = 2 * z + 1

    n_up = FermiC(i_up) * FermiA(i_up)
    n_down = FermiC(i_down) * FermiA(i_down)

    interaction = n_up * n_down - 0.5 * n_up - 0.5 * n_down + 0.25
    H_f += interaction

  H_q = jordan_wigner(H_f)

  return sum_to_hamiltonian(H_q)

def five(num_qubits):
  if num_qubits % 4 != 0:
    return None

  num_sites = num_qubits // 2
  Lx = num_sites // 2
  Ly = 2


  def site_index(jx, jy, spin):
      return 2 * ((jx - 1) * Ly + (jy - 1)) + spin

  H_f = 0.0

  for jx in range(1, Lx + 1):
      for jy in range(1, Ly + 1):
          for spin in [0, 1]:
              i = site_index(jx, jy, spin)
              k = site_index(jx + 1, jy, spin)
              hop = FermiC(i) * FermiA(k)
              hop_hc = FermiC(k) * FermiA(i)
              H_f -= (hop + hop_hc)

  for zx in range(1, Lx + 1):
    for zy in range(1, Ly + 1):
      n_up = FermiC(site_index(zx, zy, 0)) * FermiA(site_index(zx, zy, 0))
      n_down = FermiC(site_index(zx, zy, 1)) * FermiA(site_index(zx, zy, 1))

      interaction = n_up * n_down - 0.5 * n_up - 0.5 * n_down + 0.25
      H_f += interaction

  H_q = jordan_wigner(H_f)

  return sum_to_hamiltonian(H_q)


def gen_hamiltonian(label, num_qubits, j1, j2):
  if label == 0:
    return zero(num_qubits, j1, j2)
  elif label == 1:
    return one(num_qubits, j1, j2)
  elif label == 2:
    return two(num_qubits)
  elif label == 3:
    return three(num_qubits, j1, j2)
  elif label == 4:
    return four(num_qubits)
  elif label == 5:
    return five(num_qubits)
  else:
    return None