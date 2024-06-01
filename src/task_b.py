from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np

# Create a Quantum Circuit for a 2-qubit iQFT
qc = QuantumCircuit(2)

# Applying the iQFT steps
# Apply Hadamard to the second qubit
qc.h(1)
# Apply inverse Controlled-R2 gate (Controlled-phase gate with negative phase)
qc.cp(-np.pi/2, 0, 1)
# Apply Hadamard to the first qubit
qc.h(0)

# Display the circuit
print(qc.draw())

# Setup the backend
state_simulator = Aer.get_backend('statevector_simulator')

# Transpile the circuit for the backend
transpiled_qc = transpile(qc, state_simulator)

# Run the transpiled circuit on the backend
job = state_simulator.run(transpiled_qc)
result = job.result()

# Get the statevector
statevector = result.get_statevector()

# Output results
print("Statevector:", statevector)

# Since statevector calculation does not involve measurements, skip histogram plotting
