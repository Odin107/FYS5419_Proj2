import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT

def qft_matrix(n_qubits):
    """ Generates the Quantum Fourier Transform matrix for n_qubits. """
    N = 2 ** n_qubits
    omega = np.exp(2 * np.pi * 1j / N)
    return np.array([[omega ** (i * j) / np.sqrt(N) for j in range(N)] for i in range(N)])

def apply_qft(state_vector, qft_mat):
    """Applies the QFT matrix to a given state vector."""
    return np.dot(qft_mat, state_vector)

def check_unitarity(qft_mat):
    """Checks if the QFT matrix is unitary."""
    return np.allclose(np.dot(qft_mat.conj().T, qft_mat), np.eye(qft_mat.shape[0]))

def test_qft(n_qubits):
    """Generates, applies, and tests the QFT matrix for n_qubits."""
    qft_mat = qft_matrix(n_qubits)
    initial_state = np.zeros(2 ** n_qubits)
    initial_state[0] = 1  # Initial state |00...0⟩
    transformed_state = apply_qft(initial_state, qft_mat)

    print(f"QFT Matrix for {n_qubits} qubits:")
    print(qft_mat)
    print("Transformed state:", transformed_state)
    print("Unitarity check:", check_unitarity(qft_mat))

# Example usage: Test the QFT for a range of qubits
for n in range(1, 6):  # Testing from 1 to 5 qubits
    test_qft(n)
    print("\n")


def numpy_qft(n_qubits):
    """Returns the QFT matrix and transformed state using NumPy."""
    N = 2 ** n_qubits
    omega = np.exp(2 * np.pi * 1j / N)
    qft_matrix = np.array([[omega ** (i * j) / np.sqrt(N) for j in range(N)] for i in range(N)])
    initial_state = np.zeros(N)
    initial_state[0] = 1
    transformed_state = np.dot(qft_matrix, initial_state)
    return transformed_state

def qiskit_qft(n_qubits):
    """Returns the statevector after applying QFT using Qiskit."""
    qc = QuantumCircuit(n_qubits)
    qc.append(QFT(n_qubits), qc.qubits)
    statevector = Statevector.from_label('0' * n_qubits)
    return statevector.evolve(qc)

# Example usage and comparison for one, two, and three qubits
for n in range(1, 6):
    np_state = numpy_qft(n)
    qk_state = qiskit_qft(n)
    print(f"QFT for {n} qubits:")
    print("NumPy implementation state:")
    print(np_state)
    print("Qiskit implementation state:")
    print(qk_state.data)
    print("Are the states approximately equal?")
    print(np.allclose(np_state, qk_state.data))
    print("\n")
