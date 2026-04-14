import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def double_sigmoid(x, L1, k1, x01, L2, k2, x02, C):
    """2つのシグモイド曲線の和を計算する関する

    Args:
        x (_type_): _description_
        L1 (_type_): _description_
        k1 (_type_): _description_
        x01 (_type_): _description_
        L2 (_type_): _description_
        k2 (_type_): _description_
        x02 (_type_): _description_
        C (_type_): _description_
    """

    sig1 = L1 / (1 + np.exp(-k1 * (x - x01)))
    sig2 = L2 / (1 + np.exp(-k2 * (x - x02)))
    return sig1 + sig2 + C

np.random.seed(42)
x_data = np.linspace(0, 100, 200)

true_params = [10.0, 0.2, 30.0, 15.0, 0.5, 70.0, 2.0]
y_true = double_sigmoid(x_data, *true_params)
y_data = y_true + np.random.normal(0, 0.5, size=len(x_data))

initial_guess = [
    max(y_data)/2, 0.1, np.percentile(x_data, 25),
    max(y_data)/2, 0.1, np.percentile(x_data, 75),
    min(y_data)
]

popt, pcov = curve_fit(double_sigmoid, x_data, y_data, p0=initial_guess)

print("最適化されたパラメータ：")
print(f"L1（振幅1）={popt[0]:.3f}")
print(f"k1（傾き1）={popt[1]:.3f}")
print(f"x01（変曲点1）={popt[2]:.3f}")
print(f"L2（振幅2）={popt[3]:.3f}")
print(f"k2（傾き2）={popt[4]:.3f}")
print(f"x02（変曲点2）={popt[5]:.3f}")
print(f"C（ベース）={popt[6]:.3f}")

plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, label='Data with noise', color='gray', alpha=0.6, s=15)
plt.plot(x_data, double_sigmoid(x_data, *popt), label='Fitted Double Sigmoid', color='red', linewidth=2)

# それぞれのシグモイド成分を可視化（理解を深めるため）
y_comp1 = popt[0] / (1 + np.exp(-popt[1] * (x_data - popt[2]))) + popt[6]
y_comp2 = popt[3] / (1 + np.exp(-popt[4] * (x_data - popt[5]))) + popt[6]
plt.plot(x_data, y_comp1, '--', label='Component 1', color='blue')
plt.plot(x_data, y_comp2, '--', label='Component 2', color='green')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Double Sigmoid Curve Fitting')
plt.grid(True)
plt.show()