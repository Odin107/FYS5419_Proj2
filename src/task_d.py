import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

def qft_matrix(n_qubits):
    """
    Constructs the Quantum Fourier Transform (QFT) matrix for a given number of qubits.

    Parameters:
    n_qubits (int): The number of qubits.

    Returns:
    np.ndarray: The QFT matrix of size (2^n_qubits, 2^n_qubits).
    """
    N = 2 ** n_qubits
    omega = np.exp(2 * np.pi * 1j / N)
    return np.array([[omega ** (i * j) / np.sqrt(N) for j in range(N)] for i in range(N)])

def apply_qft(state_vector, qft_mat):
    """
    Applies the Quantum Fourier Transform to a given state vector using the QFT matrix.

    Parameters:
    state_vector (np.ndarray): The state vector to be transformed.
    qft_mat (np.ndarray): The QFT matrix.

    Returns:
    np.ndarray: The transformed state vector.
    """
    return np.dot(qft_mat, state_vector)

def check_unitarity(qft_mat):
    """
    Checks if a given matrix is unitary.

    Parameters:
    qft_mat (np.ndarray): The matrix to be checked for unitarity.

    Returns:
    bool: True if the matrix is unitary, False otherwise.
    """
    return np.allclose(np.dot(qft_mat.conj().T, qft_mat), np.eye(qft_mat.shape[0]))

def numpy_qft(n_qubits):
    """
    Performs the Quantum Fourier Transform using NumPy for a given number of qubits.

    Parameters:
    n_qubits (int): The number of qubits.

    Returns:
    np.ndarray: The transformed state vector after applying the QFT.
    """
    qft_mat = qft_matrix(n_qubits)
    initial_state = np.zeros(2 ** n_qubits)
    initial_state[0] = 1
    transformed_state = apply_qft(initial_state, qft_mat)
    return transformed_state

def qiskit_qft(n_qubits):
    """
    Performs the Quantum Fourier Transform using Qiskit for a given number of qubits.

    Parameters:
    n_qubits (int): The number of qubits.

    Returns:
    np.ndarray: The transformed state vector after applying the QFT using Qiskit.
    """
    qc = QuantumCircuit(n_qubits)
    qc.append(QFT(n_qubits), qc.qubits)
    statevector = Statevector.from_label('0' * n_qubits)
    return statevector.evolve(qc)

def compare_qft_results(n_qubits):
    """
    Compares the results of the Quantum Fourier Transform implemented with NumPy and Qiskit.

    Parameters:
    n_qubits (int): The number of qubits.

    Prints:
    The state vectors obtained from NumPy and Qiskit implementations, their fidelity,
    and checks the unitarity of the QFT matrix.
    """
    qft_mat = qft_matrix(n_qubits)

    # Measure execution time for NumPy implementation
    start_time = time.time()
    np_state = numpy_qft(n_qubits)
    numpy_time = time.time() - start_time

    # Measure execution time for Qiskit implementation
    start_time = time.time()
    qk_state = qiskit_qft(n_qubits).data
    qiskit_time = time.time() - start_time

    # Calculate fidelity
    fidelity = np.abs(np.vdot(np_state, qk_state)) ** 2

    # Output results
    print(f"QFT for {n_qubits} qubits:")
    print("NumPy implementation state:")
    print(np_state)
    print("Qiskit implementation state:")
    print(qk_state)
    print("Are the states approximately equal?")
    print(np.allclose(np_state, qk_state))
    print("State Fidelity with Qiskit:")
    print(fidelity)
    print("Unitarity check:", check_unitarity(qft_mat))
    print(f"Execution time for NumPy implementation: {numpy_time} seconds")
    print(f"Execution time for Qiskit implementation: {qiskit_time} seconds")

    # Visualization
    if n_qubits == 1:
        # Bloch sphere for single qubit
        fig1 = plot_bloch_multivector(np_state)
        fig1.suptitle(f"Bloch Sphere for NumPy QFT - {n_qubits} Qubit")
        fig1.savefig(f"bloch_sphere_numpy_{n_qubits}_qubit.png")

        fig2 = plot_bloch_multivector(qk_state)
        fig2.suptitle(f"Bloch Sphere for Qiskit QFT - {n_qubits} Qubit")
        fig2.savefig(f"bloch_sphere_qiskit_{n_qubits}_qubit.png")
    else:
        # Bar plot for multi-qubit
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        axes[0].bar(range(len(np_state)), np.abs(np_state)**2, color='b', alpha=0.7, label='NumPy QFT')
        axes[0].set_title(f"NumPy QFT State Probabilities for {n_qubits} Qubits")
        axes[0].set_xlabel("State Index")
        axes[0].set_ylabel("Probability")
        axes[0].legend()
        axes[1].bar(range(len(qk_state)), np.abs(qk_state)**2, color='r', alpha=0.7, label='Qiskit QFT')
        axes[1].set_title(f"Qiskit QFT State Probabilities for {n_qubits} Qubits")
        axes[1].set_xlabel("State Index")
        axes[1].set_ylabel("Probability")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"qft_state_probabilities_{n_qubits}_qubits.png")
        plt.show()

# Test the code for an arbitrary number of qubits
for n in range(1, 5):  # Test for 1 to 4 qubits
    compare_qft_results(n)
