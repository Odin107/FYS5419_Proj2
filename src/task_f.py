import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT

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
    """Convert a unitary matrix into a QuantumCircuit."""
    num_qubits = int(np.log2(len(U)))
    qc = QuantumCircuit(num_qubits)
    qc.unitary(Operator(U), range(num_qubits))
    return qc

def qpe_circuit(unitary, num_auxiliary_qubits):
    num_target_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_auxiliary_qubits + num_target_qubits, num_auxiliary_qubits)

    # Initialize auxiliary qubits in superposition
    qc.h(range(num_auxiliary_qubits))

    # Apply controlled unitary operations
    for qubit in range(num_auxiliary_qubits):
        exponent = 2 ** qubit
        controlled_U = unitary.power(exponent).control()
        qc.append(controlled_U, [qubit] + list(range(num_auxiliary_qubits, num_auxiliary_qubits + num_target_qubits)))

    # Apply Inverse QFT
    qc.append(QFT(num_auxiliary_qubits, do_swaps=False).inverse(), range(num_auxiliary_qubits))

    # Measure the auxiliary register
    qc.measure(range(num_auxiliary_qubits), range(num_auxiliary_qubits))
    return qc

# Create the unitary circuit
unitary = unitary_circuit(U)

# Create and run the QPE circuit
num_auxiliary_qubits = 4  # Adjust as needed for precision
qc = qpe_circuit(unitary, num_auxiliary_qubits)
backend = Aer.get_backend('aer_simulator')
transpiled_qc = transpile(qc, backend)
job = backend.run(transpiled_qc, shots=8192)
result = job.result()
counts = result.get_counts()

print("Measured phases:", counts)

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

# Decode the phases
decoded_phases = decode_phases(counts, num_auxiliary_qubits)

# Calculate approximate eigenvalues
eigenvalues = [2 * np.pi * phase for phase, _ in decoded_phases]

# Print the decoded phases and approximate eigenvalues
for phase, count in decoded_phases:
    eigenvalue = 2 * np.pi * phase
    print(f"Phase: {phase:.4f}, Count: {count}, Eigenvalue: {eigenvalue:.4f}")

# Save histograms for visual inspection
plot_histogram(counts).savefig("phase_estimation_histogram.png")


# Define the Hamiltonian matrix
Hx, Hz = 0.5, 0.3
H = np.array([
    [Hz, 0, 0, Hx],
    [0, -Hz, Hx, 0],
    [0, Hx, -Hz, 0],
    [Hx, 0, 0, Hz]
])

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(H)
print("Eigenvalues:", eigenvalues)
