import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 数理モデル（変更なし）
def double_sigmoid_additive(x, C, m0, L1, dm1, k1, x01, L2, dm2, k2, x02):
    base = m0 * x + C
    comp1 = (L1 + dm1 * (x - x01)) / (1 + np.exp(-k1 * (x - x01)))
    comp2 = (L2 + dm2 * (x - x02)) / (1 + np.exp(-k2 * (x - x02)))
    return base + comp1 + comp2

# 2. サンプルデータの生成（変曲点が近接している厳しい条件）
np.random.seed(42)
x_data = np.linspace(0, 100, 200)

# 真のパラメータ: x01=45, x02=55 と極めて近い
true_params = [5.0, 0.01, 
               10.0, 0.00, 0.4, 45.0, 
               12.0, 0.00, 0.6, 55.0]

y_true = double_sigmoid_additive(x_data, *true_params)
y_data = y_true + np.random.normal(0, 0.5, size=len(x_data))

# 3. フィッティングの実行（近接モデル向けのアプローチ）
n = len(x_data)

# 初期・最終ベースラインの概算
m_start, C_start = np.polyfit(x_data[:int(n*0.1)], y_data[:int(n*0.1)], 1)
m_end, C_end     = np.polyfit(x_data[-int(n*0.1):], y_data[-int(n*0.1):], 1)

# 変曲点が近い場合、全体のステップ幅(振幅)を算出
total_amplitude = np.mean(y_data[-int(n*0.1):]) - np.mean(y_data[:int(n*0.1)])

# 初期値の推論（中間プラトーの自動推論を破棄し、ドメイン知識を注入する）
# C, m0, L1, dm1, k1, x01, L2, dm2, k2, x02
initial_guess = [
    C_start, m_start,
    total_amplitude * 0.5, 0.0, 0.2, 40.0, # Comp1: 変曲点を40あたりと仮置き
    total_amplitude * 0.5, 0.0, 0.2, 60.0  # Comp2: 変曲点を60あたりと仮置き
]

# ★重要: 境界条件(Bounds)の設定
# L1, L2(振幅) は必ず正の値(0以上)とする。これで L1=1000, L2=-990 のような相殺を防ぐ。
# k1, k2(傾きの急峻さ) は正の値とする。
# x01, x02 はX軸のデータ範囲内(0〜100)とする。
lower_bounds = [-np.inf, -np.inf, 0,      -np.inf, 0.001,   0, 0,      -np.inf, 0.001,   0]
upper_bounds = [ np.inf,  np.inf, np.inf,  np.inf, np.inf, 100, np.inf, np.inf, np.inf, 100]

popt, pcov = curve_fit(
    double_sigmoid_additive, 
    x_data, y_data, 
    p0=initial_guess, 
    bounds=(lower_bounds, upper_bounds)
)

# 4. 結果の出力
C_opt, m0_opt, L1_opt, dm1_opt, k1_opt, x01_opt, L2_opt, dm2_opt, k2_opt, x02_opt = popt

print("最適化されたパラメータ:")
print(f"ベースライン: Y切片 C = {C_opt:.2f}, 初期傾き m0 = {m0_opt:.4f}")
print(f"プロセス1   : 振幅 L1 = {L1_opt:.2f}, 傾き変化量 dm1 = {dm1_opt:.4f}, 変曲点 x01 = {x01_opt:.2f}")
print(f"プロセス2   : 振幅 L2 = {L2_opt:.2f}, 傾き変化量 dm2 = {dm2_opt:.4f}, 変曲点 x02 = {x02_opt:.2f}")

# 5. 結果のプロット
plt.figure(figsize=(10, 7))

plt.scatter(x_data, y_data, label='Observed Data', color='gray', alpha=0.5, s=15)
plt.plot(x_data, double_sigmoid_additive(x_data, *popt), label='Fitted Superimposition', color='red', linewidth=2)

base = m0_opt * x_data + C_opt
comp1 = (L1_opt + dm1_opt * (x_data - x01_opt)) / (1 + np.exp(-k1_opt * (x_data - x01_opt)))
comp2 = (L2_opt + dm2_opt * (x_data - x02_opt)) / (1 + np.exp(-k2_opt * (x_data - x02_opt)))

plt.plot(x_data, base, ':', color='black', label='Baseline (Background)')
plt.plot(x_data, base + comp1, '--', color='dodgerblue', alpha=0.8, label='Base + Process 1')
plt.plot(x_data, base + comp2, '--', color='limegreen', alpha=0.8, label='Base + Process 2')

plt.axvline(x=x01_opt, color='dodgerblue', linestyle=':', alpha=0.6)
plt.axvline(x=x02_opt, color='limegreen', linestyle=':', alpha=0.6)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Fitting Two Closely Spaced Sigmoids with Bounds')
plt.grid(True)
plt.tight_layout()
plt.show()