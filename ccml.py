import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
def run_ccml(filename="source.bin", output_filename="post.bin", 
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

    # Calculate and plot probability distribution for bytes in final_output_bits
    # First, prepare the byte array as it would be saved to the file
    bits_array_for_plot = np.array(final_output_bits, dtype=np.uint8)
    if len(bits_array_for_plot) > 0:
        padded_length_for_plot = ((len(bits_array_for_plot) + 7) // 8) * 8
        padded_bits_for_plot = np.pad(bits_array_for_plot, (0, padded_length_for_plot - len(bits_array_for_plot)), constant_values=0)
        byte_array_for_plot = np.packbits(padded_bits_for_plot)

        if len(byte_array_for_plot) > 0:
            plt.figure(figsize=(12, 6))
            counts_bytes = Counter(byte_array_for_plot)
            probabilities_bytes = np.array([counts_bytes[i]/len(byte_array_for_plot) for i in range(256)])
            
            plt.bar(range(256), probabilities_bytes, color='darkslateblue', width=1.0)
            plt.title("Empirical Probability Distribution of Bytes in post.bin")
            plt.xlabel("Byte Value (0-255)")
            plt.ylabel("Probability")
            plt.xlim([-0.5, 255.5])
            if np.max(probabilities_bytes) > 0:
                 plt.ylim([0, np.max(probabilities_bytes) * 1.1])
            else:
                 plt.ylim([0, 0.1])

            # Calculate and print Shannon entropy
            entropy = 0
            for p_i in probabilities_bytes:
                if p_i > 0:
                    entropy -= p_i * np.log2(p_i)
            print(f"Shannon entropy of bytes in post.bin: {entropy:.4f} bits per symbol")

            plt.show()
        else:
            print("No byte data to plot for post.bin distribution (after packing bits).")
    else:
        print("No bit data to plot for post.bin distribution.")

    save_bits_to_binfile(final_output_bits, output_filename)

if __name__ == "__main__":
    # This part will only run when the script is executed directly
    # You can set default parameters here or parse them from command line arguments
    
    # Default input file is assumed to be 'data.bin' from the pre_processing step,
    # located one directory level up from this script.
    default_input_file = "source.bin" 
    default_output_file = "post.bin" # Output in the same directory as the script

    print(f"Running CCML with default parameters...")
    print(f"Input file: {default_input_file}")
    print(f"Output file: {default_output_file}")

    run_ccml(filename=default_input_file, 
             output_filename=default_output_file,
             N_target_bits=13000000, # example target bits
             L=8, 
             alpha=2.0, 
             epsilon=0.05, 
             omega=0.5, 
             word_bits=64)
    
    print("CCML processing complete.")

