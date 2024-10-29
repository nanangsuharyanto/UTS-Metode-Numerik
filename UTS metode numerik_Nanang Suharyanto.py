# I'll execute the provided code to see if it runs without any errors. Let's proceed with the computation.

import numpy as np
import matplotlib.pyplot as plt

# Konstanta
L = 0.5  # Induktansi dalam Henry
C = 10e-6  # Kapasitansi dalam Farad
target_f = 1000  # Frekuensi target dalam Hz
tolerance = 0.1  # Toleransi error dalam Ohm

# Fungsi untuk menghitung frekuensi resonansi f(R)
def f_R(R):
    term_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_sqrt <= 0:
        return None  # Tidak valid jika hasil akar negatif
    return (1 / (2 * np.pi)) * np.sqrt(term_sqrt)

# Turunan f(R) untuk metode Newton-Raphson
def f_prime_R(R):
    term_sqrt = 1 / (L * C) - (R**2) / (4 * L**2)
    if term_sqrt <= 0:
        return None  # Tidak valid jika hasil akar negatif
    sqrt_term = np.sqrt(term_sqrt)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Implementasi metode Newton-Raphson
def newton_raphson_method(tebakan_awal, tolerance):
    R = tebakan_awal
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None  # Kasus tidak valid
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        if f_prime_value is None:
            return None  # Kasus tidak valid
        new_R = R - f_value / f_prime_value
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Implementasi metode Bisection
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        f_mid = f_R(mid) - target_f
        if f_mid is None:
            return None  # Kasus tidak valid
        if abs(f_mid) < tolerance:
            return mid
        if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Eksekusi kedua metode
tebakan_awal = 50  # Tebakan awal untuk Newton-Raphson
interval_a, interval_b = 0, 100  # Interval Bisection

# Hasil dari Newton-Raphson
R_newton = newton_raphson_method(tebakan_awal, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Tidak ditemukan"

# Hasil dari metode Bisection
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Tidak ditemukan"

# Tampilkan hasil
print("Metode Newton-Raphson:")
print(f"Nilai R: {R_newton} ohm, Frekuensi Resonansi: {f_newton} Hz")

print("\nMetode Bisection:")
print(f"Nilai R: {R_bisection} ohm, Frekuensi Resonansi: {f_bisection} Hz")

# Plot hasil
plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="red", linestyle="--", label="Frekuensi Target 1000 Hz")

# Plot hasil Newton-Raphson
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="blue", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="blue")

# Plot hasil Bisection
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="green", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="green")

# Labeling plot
plt.xlabel("Nilai R (Ohm)")
plt.ylabel("Frekuensi Resonansi f(R) (Hz)")
plt.title("Perbandingan Metode Newton-Raphson dan Bisection")
plt.legend()
plt.grid(True)
plt.show()

# Metode Eliminasi Gauss

# Matriks koefisien dan vektor konstanta
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)

b = np.array([6, 2, 10], dtype=float)

# Implementasi eliminasi Gauss
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # Gabungkan A dan b

    # Proses eliminasi
    for i in range(n):
        for j in range(i + 1, n):
            ratio = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    # Proses substitusi balik
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Implementasi eliminasi Gauss-Jordan
def gauss_jordan(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # Gabungkan A dan b

    # Proses eliminasi
    for i in range(n):
        Ab[i] = Ab[i] / Ab[i, i]  # Buat elemen diagonal menjadi 1
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[i] * Ab[j, i]

    return Ab[:, -1]  # Solusi

# Menjalankan kedua metode eliminasi
solusi_gauss = gauss_elimination(A, b)
solusi_gauss_jordan = gauss_jordan(A, b)

# Tampilkan hasil
print("Solusi menggunakan Eliminasi Gauss:")
print(f"x1 = {solusi_gauss[0]}, x2 = {solusi_gauss[1]}, x3 = {solusi_gauss[2]}")

print("\nSolusi menggunakan Eliminasi Gauss-Jordan:")
print(f"x1 = {solusi_gauss_jordan[0]}, x2 = {solusi_gauss_jordan[1]}, x3 = {solusi_gauss_jordan[2]}")

# Analisis Error pada Metode Numerik

# Fungsi untuk menghitung R(T)
def R(T):
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Metode diferensiasi numerik

# Metode perbedaan maju
def forward_difference(T, h):
    return (R(T + h) - R(T)) / h

# Metode perbedaan mundur
def backward_difference(T, h):
    return (R(T) - R(T - h)) / h

# Metode perbedaan sentral
def central_difference(T, h):
    return (R(T + h) - R(T - h)) / (2 * h)

# Perhitungan turunan eksak
def exact_derivative(T):
    return 5000 * np.exp(3500 * (1/T - 1/298)) * (-3500 / T**2)

# Rentang suhu dan interval
temperatures = np.arange(250, 351, 10)
h = 1e-3  # Interval kecil untuk perbedaan

# Penyimpanan hasil untuk setiap metode
results = {
    "Temperature (K)": temperatures,
    "Forward Difference": [forward_difference(T, h) for T in temperatures],
    "Backward Difference": [backward_difference(T, h) for T in temperatures],
    "Central Difference": [central_difference(T, h) for T in temperatures],
    "Exact Derivative": [exact_derivative(T) for T in temperatures],
}

# Perhitungan error relatif
errors = {
    "Forward Difference Error": np.abs((np.array(results["Forward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Backward Difference Error": np.abs((np.array(results["Backward Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
    "Central Difference Error": np.abs((np.array(results["Central Difference"]) - np.array(results["Exact Derivative"])) / np.array(results["Exact Derivative"])) * 100,
}

# Plotting error relatif
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors["Backward Difference Error"], label="Backward Difference Error", marker='x')
plt.plot(temperatures, errors["Central Difference Error"], label="Central Difference Error", marker='s')

# Labeling the plot
plt.xlabel("Temperature (K)")
plt.ylabel("Relative Error (%)")
plt.title("Relative Error in Numerical Differentiation Methods")
plt.legend()
plt.grid(True)
plt.show()
