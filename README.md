# Skrypt Post-Processingu CCML (ccml.py)

## Opis

Skrypt `ccml.py` implementuje algorytm Chaotycznie Sprzężonej Sieci Map (Chaotically Coupled Map Lattice - CCML) w celu post-processingu sekwencji bitów. Celem jest poprawa właściwości statystycznych i losowości wejściowego strumienia bitów, często pochodzącego z fizycznego generatora liczb losowych (TRNG).

Skrypt wykonuje następujące główne kroki:
1.  Wczytuje wejściowy strumień bitów z pliku binarnego (np. `source.bin`).
2.  Inicjalizuje stany sieci CCML.
3.  W pętli iteracyjnej:
    a.  Zaburza (perturbuje) stany sieci CCML przy użyciu fragmentów wejściowego strumienia bitów.
    b.  Ewoluuje stany sieci zgodnie z dynamiką CCML (mapa odcinkowo-liniowa z dyfuzją).
    c.  Konwertuje zaktualizowane stany na liczby całkowite (`z_array`).
    d.  Wykonuje operację zamiany połówek bitów (bit-swap) na elementach `z_array`.
    e.  Ekstrahuje bity z przetworzonych wartości `z_array` i dodaje je do wyjściowego strumienia.
4.  Kontynuuje proces, aż do zebrania docelowej liczby bitów wyjściowych.
5.  Zapisuje przetworzony strumień bitów do pliku binarnego (np. `post.bin`).

## Zależności

Do uruchomienia skryptu wymagana jest następująca biblioteka Python:

*   `numpy`

Można ją zainstalować za pomocą pip:
```bash
pip install numpy