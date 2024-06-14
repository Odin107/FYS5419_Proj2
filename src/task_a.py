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

"""
example output:
QFT Matrix for 1 qubits:
[[ 0.70710678+0.00000000e+00j  0.70710678+0.00000000e+00j]
 [ 0.70710678+0.00000000e+00j -0.70710678+8.65956056e-17j]]

Check unitarity: True

Transform of |00...0⟩ state: [0.70710678+0.j 0.70710678+0.j]


QFT Matrix for 2 qubits:
[[ 5.00000000e-01+0.0000000e+00j  5.00000000e-01+0.0000000e+00j
   5.00000000e-01+0.0000000e+00j  5.00000000e-01+0.0000000e+00j]
 [ 5.00000000e-01+0.0000000e+00j  3.06161700e-17+5.0000000e-01j
  -5.00000000e-01+6.1232340e-17j -9.18485099e-17-5.0000000e-01j]
 [ 5.00000000e-01+0.0000000e+00j -5.00000000e-01+6.1232340e-17j
   5.00000000e-01-1.2246468e-16j -5.00000000e-01+1.8369702e-16j]
 [ 5.00000000e-01+0.0000000e+00j -9.18485099e-17-5.0000000e-01j
  -5.00000000e-01+1.8369702e-16j  2.75545530e-16+5.0000000e-01j]]

Check unitarity: True

Transform of |00...0⟩ state: [0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j]


QFT Matrix for 3 qubits:
[[ 3.53553391e-01+0.00000000e+00j  3.53553391e-01+0.00000000e+00j
   3.53553391e-01+0.00000000e+00j  3.53553391e-01+0.00000000e+00j
   3.53553391e-01+0.00000000e+00j  3.53553391e-01+0.00000000e+00j
   3.53553391e-01+0.00000000e+00j  3.53553391e-01+0.00000000e+00j]
 [ 3.53553391e-01+0.00000000e+00j  2.50000000e-01+2.50000000e-01j
   7.85046229e-17+3.53553391e-01j -2.50000000e-01+2.50000000e-01j
  -3.53553391e-01+1.57009246e-16j -2.50000000e-01-2.50000000e-01j
  -2.35513869e-16-3.53553391e-01j  2.50000000e-01-2.50000000e-01j]
 [ 3.53553391e-01+0.00000000e+00j  7.85046229e-17+3.53553391e-01j
  -3.53553391e-01+1.57009246e-16j -2.35513869e-16-3.53553391e-01j
   3.53553391e-01-3.14018492e-16j  3.92523115e-16+3.53553391e-01j
  -3.53553391e-01+4.71027738e-16j -5.49532361e-16-3.53553391e-01j]
 [ 3.53553391e-01+0.00000000e+00j -2.50000000e-01+2.50000000e-01j
  -2.35513869e-16-3.53553391e-01j  2.50000000e-01+2.50000000e-01j
  -3.53553391e-01+4.71027738e-16j  2.50000000e-01-2.50000000e-01j
   7.06541606e-16+3.53553391e-01j -2.50000000e-01-2.50000000e-01j]
 [ 3.53553391e-01+0.00000000e+00j -3.53553391e-01+1.57009246e-16j
   3.53553391e-01-3.14018492e-16j -3.53553391e-01+4.71027738e-16j
   3.53553391e-01-6.28036983e-16j -3.53553391e-01+7.85046229e-16j
   3.53553391e-01-9.42055475e-16j -3.53553391e-01+1.09906472e-15j]
 [ 3.53553391e-01+0.00000000e+00j -2.50000000e-01-2.50000000e-01j
   3.92523115e-16+3.53553391e-01j  2.50000000e-01-2.50000000e-01j
  -3.53553391e-01+7.85046229e-16j  2.50000000e-01+2.50000000e-01j
  -1.17756934e-15-3.53553391e-01j -2.50000000e-01+2.50000000e-01j]
 [ 3.53553391e-01+0.00000000e+00j -2.35513869e-16-3.53553391e-01j
  -3.53553391e-01+4.71027738e-16j  7.06541606e-16+3.53553391e-01j
   3.53553391e-01-9.42055475e-16j -1.17756934e-15-3.53553391e-01j
  -3.53553391e-01+1.41308321e-15j  1.64859708e-15+3.53553391e-01j]
 [ 3.53553391e-01+0.00000000e+00j  2.50000000e-01-2.50000000e-01j
  -5.49532361e-16-3.53553391e-01j -2.50000000e-01-2.50000000e-01j
  -3.53553391e-01+1.09906472e-15j -2.50000000e-01+2.50000000e-01j
   1.64859708e-15+3.53553391e-01j  2.50000000e-01+2.50000000e-01j]]

Check unitarity: True

Transform of |00...0⟩ state: [0.35355339+0.j 0.35355339+0.j 0.35355339+0.j 0.35355339+0.j
 0.35355339+0.j 0.35355339+0.j 0.35355339+0.j 0.35355339+0.j]

"""