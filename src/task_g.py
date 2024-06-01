import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from math import gcd


def shors_algorithm(N, a):
    def modular_exponentiation(qc, a, N, n_count):
        # Implement modular exponentiation a^x mod N using the quantum circuit
        for i in range(n_count):
            control_qubit = i
            target_qubit = n_count + (i % 4)  # Ensure target_qubit index is within range
            qc.cp(2 * np.pi / (2**i), control_qubit, target_qubit)  # Example placeholder using cp (controlled phase)

    def qpe_amodN(a, N):
        n_count = 8  # Number of counting qubits
        total_qubits = n_count + 4  # Total qubits including the four auxiliary qubits
        qc = QuantumCircuit(total_qubits, n_count)
        for q in range(n_count):
            qc.h(q)
        qc.x(3 + n_count)

        for q in range(n_count):
            modular_exponentiation(qc, a**(2**q) % N, N, n_count)
        
        qc.append(QFT(n_count, do_swaps=False).inverse(), range(n_count))
        qc.measure(range(n_count), range(n_count))
        return qc

    # Quantum Phase Estimation Circuit
    qpe_circuit = qpe_amodN(a, N)
    simulator = Aer.get_backend('aer_simulator')
    transpiled_qpe = transpile(qpe_circuit, simulator)
    job = simulator.run(transpiled_qpe, shots=2048)
    result = job.result()
    counts = result.get_counts()

    # Print the measurement results
    print("Measured phases:", counts)
    
    # Save and display the histogram
    histogram = plot_histogram(counts)
    histogram.savefig("phase_estimation_histogram.png")
    histogram.show()
    
    return counts

N = 15  # Number to factor
a = 7   # Random number chosen
counts = shors_algorithm(N, a)

def binary_to_decimal(binary_string, num_auxiliary_qubits):
    """Convert a binary string to a decimal fraction."""
    return int(binary_string, 2) / (2 ** num_auxiliary_qubits)

def decode_phases(counts, num_auxiliary_qubits):
    """Decode the measured phases from the QPE circuit."""
    phases = []
    for binary_phase, count in counts.items():
        decimal_phase = binary_to_decimal(binary_phase, num_auxiliary_qubits)
        phases.append((decimal_phase, count))
    return phases

# Number of auxiliary qubits used
num_auxiliary_qubits = 8

# Decode the phases
decoded_phases = decode_phases(counts, num_auxiliary_qubits)

# Calculate approximate eigenvalues
eigenvalues = [2 * np.pi * phase for phase, _ in decoded_phases]

# Print the decoded phases and approximate eigenvalues
for phase, count in decoded_phases:
    eigenvalue = 2 * np.pi * phase
    print(f"Phase: {phase:.4f}, Count: {count}, Eigenvalue: {eigenvalue:.4f}")

# Find the period
def find_period(N, a):
    for r in range(1, N):
        if pow(a, r, N) == 1:
            return r
    return None

# Find the factors using the period
def find_factors(N, r, a):
    if r is None:
        return None
    factor1 = gcd(a**(r//2) - 1, N)
    factor2 = gcd(a**(r//2) + 1, N)
    if factor1 == 1 or factor1 == N:
        return None
    if factor2 == 1 or factor2 == N:
        return None
    return factor1, factor2

# Example values
N = 15
a = 7

# Find the period
r = find_period(N, a)
print(f"Period r: {r}")

# Find the factors using the period
factors = find_factors(N, r, a)
print(f"Factors: {factors}")
