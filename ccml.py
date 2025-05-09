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
    val = int(val)
    left_mask = (1 << L_half) - 1
    right_mask = (1 << (int(np.ceil(val.bit_length() / L_half)) * L_half - L_half)) - 1
    word_bits = 64

    left_val = (val >> L_half) & ((1 << (word_bits - L_half)) - 1)
    right_val = val & ((1 << L_half) - 1)
    return (right_val << (word_bits - L_half)) | left_val

# Inicjalizacja x0j
def initialize_states(L):
    ps_values = [0.141592, 0.653589, 0.793238, 0.462643,
                 0.383279, 0.502884]
    if L == 8:
        ps_values.extend([0.197169, 0.399375])
    
    if L <= len(ps_values):
        return np.array(ps_values[:L])
    else:
        base_fallback = [0.141592, 0.271828, 0.173205, 0.707106,
                         0.577215, 0.693147, 0.301030, 0.414213]
        return np.array((ps_values + base_fallback * (L // len(base_fallback) + 1))[:L])

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

# Główna funkcja CCML
def run_ccml(filename="input.bin", output_filename="output.bin", 
             N_target_bits=1024, 
             L=8, alpha=2.0, epsilon=0.05, omega=0.5, word_bits=64):
    
    input_all_bits = read_bits_from_binfile(filename)
    
    gamma = L // 2 
    b_perturb = 3

    states = initialize_states(L)
    output_bits_collected = []
    input_bits_cursor = 0

    while len(output_bits_collected) < N_target_bits:
        if input_bits_cursor + L * b_perturb > len(input_all_bits):
            print("Not enough input bits for a full perturbation round.")
            break 

        current_states_before_perturb = np.copy(states)
        for j_perturb in range(L):
            yc_bits_list = input_all_bits[input_bits_cursor : input_bits_cursor + b_perturb]
            input_bits_cursor += b_perturb
            
            if len(yc_bits_list) < b_perturb:
                print("Error: Ran out of bits for yc_bits_list.")
                return

            yc_int = int("".join(str(b) for b in yc_bits_list), 2)

            term_yc_numerator = omega * yc_int
            term_yc_denominator = (2**b_perturb - 1)
            
            if term_yc_denominator == 0: term_yc_denominator = 1 

            term_yc = term_yc_numerator / term_yc_denominator
            
            states[j_perturb] = (term_yc + current_states_before_perturb[j_perturb]) / (1 + omega)
            states[j_perturb] = np.clip(states[j_perturb], 0.0, 1.0)

        for _ in range(gamma):
            states = ccml_step(states, alpha, epsilon)

        z_array = np.zeros(L, dtype=np.uint64)
        for j_z_conversion in range(L):
            if states[j_z_conversion] >= 1.0:
                 z_array[j_z_conversion] = np.uint64(2**word_bits - 1)
            else:
                 z_array[j_z_conversion] = np.uint64(np.floor(states[j_z_conversion] * (2**word_bits)))

        for j_swap in range(L // 2):
            idx_second_half = j_swap + (L // 2)
            if idx_second_half < L:
                 swapped_half_val = bit_swap64(z_array[idx_second_half], word_bits // 2)
                 z_array[j_swap] = z_array[j_swap] ^ swapped_half_val
            else:
                 print(f"Warning: index {idx_second_half} out of bounds for z_array in swap.")

        for j_concat in range(L // 2):
            if len(output_bits_collected) >= N_target_bits:
                break
            bits_from_z = format(z_array[j_concat], f'0{word_bits}b')
            output_bits_collected.extend([int(b) for b in bits_from_z])
        
        if len(output_bits_collected) >= N_target_bits:
            break

    final_output_bits = output_bits_collected[:N_target_bits]
    save_bits_to_binfile(final_output_bits, output_filename)
    print(f"Zapisano {len(final_output_bits)} bitów do pliku {output_filename}")

