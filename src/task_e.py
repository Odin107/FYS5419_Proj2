import numpy as np
from numpy import pi, log, ceil
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def qft_rot(qc, n):
    """
    Apply Quantum Fourier Transform (QFT) rotation gates to a quantum circuit.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit to apply the QFT rotation gates to.
    - n (int): The number of qubits to apply the QFT rotation gates to.

    Returns:
    - qc (QuantumCircuit): The quantum circuit with the QFT rotation gates applied.
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
    Swaps the qubits in a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to perform the qubit swap on.
        n (int): The number of qubits in the circuit.

    Returns:
        QuantumCircuit: The quantum circuit with the qubits swapped.
    """
    for qb in range(n//2):
        qc.swap(qb, n-1-qb)
    return qc

def QFT(n):
    """
    Applies the Quantum Fourier Transform (QFT) on an n-qubit quantum circuit.

    Args:
        n (int): The number of qubits in the circuit.

    Returns:
        Gate: The QFT gate representing the transformation.

    """
    qc = QuantumCircuit(n, name='QFT')
    qft_rot(qc, n)
    swap_qb(qc, n)
    gate = qc.to_gate()
    return gate

def IQFT(n):
    """
    Performs the Inverse Quantum Fourier Transform (IQFT) on n qubits.

    Args:
        n (int): The number of qubits.

    Returns:
        Gate: The IQFT gate.

    """
    qc = QuantumCircuit(n, name='IQFT')
    swap_qb(qc, n)
    iqft_rot(qc, 0, n)
    gate = qc.to_gate()
    return gate

def qpe(n_result_qb, phase):
    """
    Implements the Quantum Phase Estimation algorithm.

    Args:
        n_result_qb (int): The number of qubits used to store the estimated phase.
        phase (float): The phase to be estimated.

    Returns:
        QuantumCircuit: The quantum circuit representing the Quantum Phase Estimation algorithm.
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
    Simulates the given quantum circuit and returns the probability distribution of the measurement outcomes.

    Args:
        qc (QuantumCircuit): The quantum circuit to be simulated.
        shots (int): The number of times the circuit should be executed.

    Returns:
        dict: A dictionary representing the probability distribution of the measurement outcomes.
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
    Evaluates the phase estimation algorithm for a given phase.

    Args:
        phase (float): The phase to be estimated.
        accuracy (int): The number of accurate bits in the estimation.
        shots (int, optional): The number of shots for the simulation. Defaults to 2048.
        success_prob (float, optional): The desired success probability. Defaults to 0.9.

    Returns:
        tuple: A tuple containing the following:
            - prob (dict): A dictionary of the simulated probabilities for each binary key.
            - key_max (str): The binary key with the maximum probability.
            - prob_max (float): The maximum probability.
            - p (float): The actual success probability.
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