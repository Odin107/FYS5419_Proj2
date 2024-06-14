import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
#from qiskit.circuit.library import QFT
from numpy import pi, log, ceil
import matplotlib.pyplot as plt


# Define the parameters
E1, E2 = 1.0, 2.0  # Energy levels
Hx, Hz = 0.5, 0.3  # Interaction strengths

# Non-interacting Hamiltonian (H0)
H0 = np.array([[E1, 0], 
               [0, E2]])

# Interaction Hamiltonian (HI)
sigma_x = np.array([[0, 1], 
                    [1, 0]])
sigma_z = np.array([[1, 0], 
                    [0, -1]])

# Construct HI using tensor products
HI = Hx * np.kron(sigma_x, sigma_x) + Hz * np.kron(sigma_z, sigma_z)

# Total Hamiltonian for two qubits
H = np.kron(np.eye(2), H0) + HI

# Time parameter for the unitary evolution
t = np.pi  # Chosen for simplicity; may need adjustment based on the specific problem

# Create the unitary operator from H
U = expm(1j * H * t)


def unitary_circuit(U):
    """
    Creates a quantum circuit for applying a unitary operator U.

    Parameters:
    U (array-like): The unitary operator to be applied.

    Returns:
    QuantumCircuit: The quantum circuit with the unitary operator applied.
    """
    num_qubits = int(np.log2(len(U)))
    qc = QuantumCircuit(num_qubits)
    qc.unitary(Operator(U), range(num_qubits))
    return qc


def qpe_circuit(unitary, num_auxiliary_qubits):
    """
    Constructs a quantum phase estimation circuit.

    Args:
        unitary (QuantumCircuit): The unitary operator for which the phase is to be estimated.
        num_auxiliary_qubits (int): The number of auxiliary qubits used in the circuit.

    Returns:
        QuantumCircuit: The quantum phase estimation circuit.
    """
    num_target_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_auxiliary_qubits + num_target_qubits, num_auxiliary_qubits)
    qc.h(range(num_auxiliary_qubits))
    for qubit in range(num_auxiliary_qubits):
        exponent = 2 ** qubit
        controlled_U = unitary.power(exponent).control()
        qc.append(controlled_U, [qubit] + list(range(num_auxiliary_qubits, num_auxiliary_qubits + num_target_qubits)))
    qc.append(QFT(num_auxiliary_qubits, do_swaps=False).inverse(), range(num_auxiliary_qubits))
    qc.measure(range(num_auxiliary_qubits), range(num_auxiliary_qubits))
    return qc


def qft_rot(qc, n):
    """
    Apply Quantum Fourier Transform (QFT) rotation gates to a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to apply the QFT rotation gates to.
        n (int): The number of qubits to apply the QFT rotation gates on.

    Returns:
        QuantumCircuit: The quantum circuit with the QFT rotation gates applied.
    """
    if n == 0: return qc
    n -= 1
    qc.h(n)
    for qb in range(n):
        qc.cp(pi/2**(n-qb), qb, n)
    qft_rot(qc, n)


def iqft_rot(qc, n, nbits):
    """
    Apply the inverse quantum Fourier transform (IQFT) rotation gates to the given quantum circuit.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit to apply the IQFT rotation gates to.
    - n (int): The current index of the qubit being operated on.
    - nbits (int): The total number of qubits in the circuit.

    Returns:
    - qc (QuantumCircuit): The quantum circuit with the IQFT rotation gates applied.
    """
    if n == nbits: return qc
    for qb in range(n):
        qc.cp(-pi/2**(n-qb), qb, n)
    qc.h(n)
    n += 1
    iqft_rot(qc, n, nbits)


def swap_qb(qc, n):
    """
    Swap qubits in a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        n (int): The number of qubits in the circuit.

    Returns:
        QuantumCircuit: The modified quantum circuit with swapped qubits.
    """
    for qb in range(n//2):
        qc.swap(qb, n-1-qb)
    return qc


def QFT(n):
    """
    Create a Quantum Fourier Transform (QFT) gate for a specified number of qubits.

    Args:
        n (int): The number of qubits for the QFT gate.

    Returns:
        Gate: The QFT gate for the given number of qubits.
    """
    qc = QuantumCircuit(n, name='QFT')
    qft_rot(qc, n)
    swap_qb(qc, n)
    gate = qc.to_gate()
    return gate


def IQFT(n):
    """
    Create an Inverse Quantum Fourier Transform (IQFT) gate for a specified number of qubits.

    Args:
        n (int): The number of qubits for the IQFT gate.

    Returns:
        Gate: The IQFT gate for the given number of qubits.
    """
    qc = QuantumCircuit(n, name='IQFT')
    swap_qb(qc, n)
    iqft_rot(qc, 0, n)
    gate = qc.to_gate()
    return gate


def qpe(n_result_qb, phase):
    """
    Create a quantum phase estimation circuit.

    Args:
        n_result_qb (int): The number of qubits to use for the result register.
        phase (float): The phase to estimate.

    Returns:
        QuantumCircuit: The quantum phase estimation circuit.
    """
    angle = 2*pi*phase
    t = n_result_qb
    qc = QuantumCircuit(t+1, t)
    qc.x(t)
    qc.h(range(t))
    for bit in range(t):
        for i in range(2**(bit)):
            qc.cp(angle, bit, t)
    qc.barrier()
    qc.append(IQFT(t), range(t))
    qc.measure(range(t), range(t))
    return qc


def simulate(qc, shots):
    """
    Simulate a quantum circuit and plot the result histogram.

    Args:
        qc (QuantumCircuit): The quantum circuit to simulate.
        shots (int): The number of shots for the simulation.

    Returns:
        dict: The probability distribution of the measurement results.
    """
    sim = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, sim)
    result = sim.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    
    total_shots = sum(counts.values())
    prob = {k: v / total_shots for k, v in counts.items()}
    
    fig = plt.figure()
    plot_histogram(prob)
    plt.savefig("qpe_histogram.png")
    return prob


def evaluate_phase_estimation(phase, accuracy, shots=2048, success_prob=0.9):
    """
    Evaluate the quantum phase estimation algorithm.

    Args:
        phase (float): The phase to estimate.
        accuracy (int): The number of bits for accuracy in the estimation.
        shots (int, optional): The number of shots for the simulation. Default is 2048.
        success_prob (float, optional): The desired success probability. Default is 0.9.

    Returns:
        tuple: A tuple containing the probability distribution, key with maximum probability, maximum probability, and actual success probability.
    """
    e = 1 - success_prob
    t = accuracy + ceil(log(2 + 1/(2*e)) / log(2))
    t = int(t)

    qc = qpe(t, phase)
    prob = simulate(qc, shots)

    prob_max = 0
    for key in prob.keys():
        if prob[key] > prob_max:
            prob_max = prob[key]
            key_max = key

    print("pmax(", key_max, ")=", prob_max)

    phi_tmp = phase
    val = 0
    for i in range(accuracy):
        val = 2*val + int(2*phi_tmp)
        phi_tmp = 2*phi_tmp - int(2*phi_tmp)

    p = 0
    for key in prob.keys():
        if (int(key, 2) >> (t-accuracy)) == val:
            print(key, "-->", prob[key])
            p += prob[key]

    print("Actual success probability:", p)
    print("Theoretical lower bound on success probability:", success_prob)

    if p < success_prob:
        print("Something is wrong")

    # Correctly extract phase from binary result
    extracted_phase = int(key_max, 2) / (2**len(key_max))
    
    # Normalize extracted phase
    extracted_phase = extracted_phase % 1
    
    print("Extracted phase:", extracted_phase)
    print("Actual phase:", phase)

    return prob, key_max, prob_max, p

# Define the phase and desired accuracy
phase = 1/3
accuracy = 13  # Adjusted to match the key length from the output

# Evaluate phase estimation
prob, key_max, prob_max, p = evaluate_phase_estimation(phase, accuracy)

# Decode phases and calculate eigenvalues
decoded_phases = [(int(key, 2) / (2**len(key)), prob[key]) for key in prob.keys()]
extracted_eigenvalues = [2 * pi * phase for phase, _ in decoded_phases]

# Print decoded phases and extracted eigenvalues
for phase, count in decoded_phases:
    eigenvalue = 2 * pi * phase
    print(f"Phase: {phase:.4f}, Count: {count}, Eigenvalue: {eigenvalue:.4f}")

# Calculate the eigenvalues from Hamiltonian
theoretical_eigenvalues = np.linalg.eigvals(H)
print("Theoretical Eigenvalues:", theoretical_eigenvalues)

# Plot extracted eigenvalues vs theoretical eigenvalues
plt.figure()
plt.scatter(range(len(extracted_eigenvalues)), extracted_eigenvalues, color='blue', label='Extracted Eigenvalues')
plt.scatter(range(len(theoretical_eigenvalues)), theoretical_eigenvalues, color='red', label='Theoretical Eigenvalues', marker='x')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Comparison of Extracted and Theoretical Eigenvalues')
plt.legend()
plt.grid(True)
plt.savefig("eigenvalues_comparison.png")
plt.show()
