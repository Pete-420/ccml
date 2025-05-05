import numpy as np

# Tent-like chaotic map
def piecewise_map(x, alpha=2.0):
    return alpha * x if x <= 0.5 else alpha * (1 - x)

# Jedna iteracja CCML
def ccml_step(states, alpha=2.0, epsilon=0.05):
    L = len(states)
    new_states = np.zeros_like(states)
    for i in range(L):
        left = piecewise_map(states[(i - 1) % L], alpha)
        center = piecewise_map(states[i], alpha)
        right = piecewise_map(states[(i + 1) % L], alpha)

        new_states[i] = (
            (1 - epsilon) * center +
            (epsilon / 2.0) * (left + right)
        )
    return new_states

# Konwersja bitów na float ∈ (0,1)
def bits_to_floats(bits, group_size=8):
    floats = []
    for i in range(0, len(bits) - group_size + 1, group_size):
        byte = bits[i:i + group_size]
        int_val = int("".join(str(b) for b in byte), 2)
        floats.append((int_val + 0.5) / (2 ** group_size))
    return floats

# Wczytanie pliku binarnego jako lista bitów
def read_bits_from_binfile(filename):
    with open(filename, "rb") as f:
        byte_data = f.read()
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    return bits.tolist()

# Zapis bitów do pliku .bin
def save_bits_to_binfile(bits, output_filename):
    bits_array = np.array(bits, dtype=np.uint8)
    padded_length = ((len(bits_array) + 7) // 8) * 8
    padded_bits = np.pad(bits_array, (0, padded_length - len(bits_array)), constant_values=0)
    byte_array = np.packbits(padded_bits)
    with open(output_filename, "wb") as f:
        f.write(byte_array)

# Główna funkcja CCML z zapisem do pliku
def run_ccml_from_binfile(filename="input.bin", output_filename="output.bin", L=64, alpha=2.0, epsilon=0.05, iterations=2):
    bits = read_bits_from_binfile(filename)
    floats = bits_to_floats(bits, group_size=8)

    states = np.array(floats[:L])
    output_bits = []

    i = L
    while i + L <= len(floats):
        perturb = np.array(floats[i:i+L])
        states = (states + perturb) % 1.0

        for _ in range(iterations):
            states = ccml_step(states, alpha, epsilon)

        for val in states:
            b = int(val * 256) % 256
            output_bits.extend([int(x) for x in f"{b:08b}"])
        
        i += L

    save_bits_to_binfile(output_bits, output_filename)
    print(f"Zapisano {len(output_bits)} bitów do pliku: {output_filename}")
