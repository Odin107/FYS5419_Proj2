import numpy as np


def qft_matrix(n_qubits):
    """
    Generates the Quantum Fourier Transform matrix for n_qubits.

    Parameters:
    - n_qubits (int): The number of qubits.

    Returns:
    - qft_matrix (ndarray): The Quantum Fourier Transform matrix of shape (2^n_qubits, 2^n_qubits).
    """
    N = 2 ** n_qubits
    omega = np.exp(2 * np.pi * 1j / N)
    return np.array([[omega ** (i * j) / np.sqrt(N) for j in range(N)] for i in range(N)])


def check_unitarity(qft_matrix):
    """
    Checks the unitarity of the given QFT matrix.

    Parameters:
    qft_matrix (numpy.ndarray): The QFT matrix to be checked.

    Returns:
    bool: True if the QFT matrix is unitary, False otherwise.
    """
    return np.allclose(np.dot(qft_matrix.conj().T, qft_matrix), np.eye(qft_matrix.shape[0]))


def initial_state_transform(qft_matrix):
    """
    Computes the Quantum Fourier Transform (QFT) of the initial state |00...0⟩.

    Parameters:
    qft_matrix (numpy.ndarray): The QFT matrix used for the transformation.

    Returns:
    numpy.ndarray: The transformed state after applying the QFT.
    """
    initial_state = np.zeros(qft_matrix.shape[0])
    initial_state[0] = 1
    return np.dot(qft_matrix, initial_state)


for n in range(1, 4):
    qft_mat = qft_matrix(n)
    print(f"QFT Matrix for {n} qubits:")
    print(qft_mat)
    print("Check unitarity:", check_unitarity(qft_mat))
    print("Transform of |00...0⟩ state:", initial_state_transform(qft_mat))
    print("\n")
