import numpy as np

# Tent-like chaotic map
def piecewise_map(x, alpha=2.0):
    return alpha * x if x <= 0.5 else alpha * (1 - x)

# CCML krok
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

# Swap polówek bitów 64-bitowej liczby
def bit_swap64(val, L_half):
    left = (val >> L_half) & ((1 << (64 - L_half)) - 1)
    right = val & ((1 << L_half) - 1)
    return (right << (64 - L_half)) | left

# Inicjalizacja x0j
def initialize_states(L):
    base_values = [0.141592, 0.271828, 0.173205, 0.707106,
                   0.577215, 0.693147, 0.301030, 0.414213]
    return np.array(base_values[:L])

# Wczytanie pliku binarnego jako lista bitów
def read_bits_from_binfile(filename):
    with open(filename, "rb") as f:
        byte_data = f.read()
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    return bits.tolist()

# Zapis bitów do pliku binarnego
def save_bits_to_binfile(bits, output_filename):
    bits_array = np.array(bits, dtype=np.uint8)
    padded_length = ((len(bits_array) + 7) // 8) * 8
    padded_bits = np.pad(bits_array, (0, padded_length - len(bits_array)), constant_values=0)
    byte_array = np.packbits(padded_bits)
    with open(output_filename, "wb") as f:
        f.write(byte_array)

# Zamiana bitów na float (0,1)
def bits_to_floats(bits, group_size=8):
    floats = []
    for i in range(0, len(bits) - group_size + 1, group_size):
        byte = bits[i:i + group_size]
        int_val = int("".join(str(b) for b in byte), 2)
        floats.append((int_val + 0.5) / (2 ** group_size))
    return floats

# Główna funkcja CCML

def run_ccml(filename="input.bin", output_filename="output.bin", L=8, alpha=2.0, epsilon=0.05, omega=0.5, word_bits=64):
    bits = read_bits_from_binfile(filename)
    floats = bits_to_floats(bits, group_size=8)

    gamma = max(1, int(np.floor(L**2 / (2**5))))  # uproszczona wersja
    scale = (1 + omega) / (2**8 - 1)

    states = initialize_states(L)
    output_values = []

    i = 0
    while i + L <= len(floats):
        perturb = np.array(floats[i:i+L])
        states = (omega * perturb + states) * scale

        for _ in range(gamma):
            states = ccml_step(states, alpha, epsilon)

        # Eq. (5): uint64 z zamianą bitów
        for val in states:
            z = int(val * (2**word_bits)) % 2**word_bits
            z_swapped = bit_swap64(z, word_bits // 2)
            output_values.append(z_swapped)

        i += L

    # Konwersja do bitów i zapis
    output_bits = []
    for val in output_values:
        output_bits.extend([int(b) for b in f"{val:0{word_bits}b}"])

    save_bits_to_binfile(output_bits, output_filename)
    print(f"Zapisano {len(output_bits)} bitów do pliku {output_filename}")

