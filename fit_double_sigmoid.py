import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 数理モデルの定義（完全な重ね合わせモデル）
def double_sigmoid_additive(x, C, m0, L1, dm1, k1, x01, L2, dm2, k2, x02):
    """
    ベースラインに対して、独立した2つのシグモイドプロセス（それぞれが傾きの変化を伴う）を加算する関数
    """
    # ベースライン（初期のドリフト）
    base = m0 * x + C
    
    # コンポーネント1（振幅L1と、発生による傾き変化dm1を持つ）
    comp1 = (L1 + dm1 * (x - x01)) / (1 + np.exp(-k1 * (x - x01)))
    
    # コンポーネント2（振幅L2と、発生による傾き変化dm2を持つ）
    comp2 = (L2 + dm2 * (x - x02)) / (1 + np.exp(-k2 * (x - x02)))
    
    return base + comp1 + comp2

# 2. サンプルデータの生成
np.random.seed(42)
x_data = np.linspace(0, 100, 200)

# 真のパラメータ
# ベース: 切片=5, 初期傾き=-0.05
# Comp1: 振幅=10, 傾き変化=+0.05 (現象1の後は傾き0になる)
# Comp2: 振幅=15, 傾き変化=+0.10 (現象2の後は傾き0.1になる)
true_params = [5.0, -0.05, 
               10.0, 0.05, 0.3, 30.0, 
               15.0, 0.10, 0.5, 70.0]

y_true = double_sigmoid_additive(x_data, *true_params)
y_data = y_true + np.random.normal(0, 0.5, size=len(x_data))

# 3. フィッティングの実行（初期値の自動推論）
n = len(x_data)
# 領域ごとの傾きの概算
m_start, C_start = np.polyfit(x_data[:int(n*0.1)], y_data[:int(n*0.1)], 1)
m_mid, C_mid     = np.polyfit(x_data[int(n*0.45):int(n*0.55)], y_data[int(n*0.45):int(n*0.55)], 1)
m_end, C_end     = np.polyfit(x_data[-int(n*0.1):], y_data[-int(n*0.1):], 1)

# 傾きの変化量(dm)と振幅(L)の初期値推定
m0_g = m_start
dm1_g = m_mid - m_start
dm2_g = m_end - m_mid

L1_g = np.mean(y_data[int(n*0.45):int(n*0.55)]) - np.mean(y_data[:int(n*0.1)])
L2_g = np.mean(y_data[-int(n*0.1):]) - np.mean(y_data[int(n*0.45):int(n*0.55)])

initial_guess = [
    C_start, m0_g,
    L1_g, dm1_g, 0.1, np.percentile(x_data, 25), # Comp1
    L2_g, dm2_g, 0.1, np.percentile(x_data, 75)  # Comp2
]

# パラメータの最適化
popt, pcov = curve_fit(double_sigmoid_additive, x_data, y_data, p0=initial_guess)

# 4. 結果の出力
C_opt, m0_opt, L1_opt, dm1_opt, k1_opt, x01_opt, L2_opt, dm2_opt, k2_opt, x02_opt = popt

print("最適化されたパラメータ:")
print(f"ベースライン: Y切片 C = {C_opt:.2f}, 初期傾き m0 = {m0_opt:.4f}")
print(f"プロセス1   : 振幅 L1 = {L1_opt:.2f}, 傾き変化量 dm1 = {dm1_opt:.4f}, 変曲点 x01 = {x01_opt:.2f}")
print(f"プロセス2   : 振幅 L2 = {L2_opt:.2f}, 傾き変化量 dm2 = {dm2_opt:.4f}, 変曲点 x02 = {x02_opt:.2f}")

# 5. 結果のプロット
plt.figure(figsize=(10, 7))

# 実データと全体のフィット
plt.scatter(x_data, y_data, label='Observed Data', color='gray', alpha=0.5, s=15)
plt.plot(x_data, double_sigmoid_additive(x_data, *popt), label='Fitted Superimposition', color='red', linewidth=2)

# 各コンポーネントを分解して計算
base = m0_opt * x_data + C_opt
comp1 = (L1_opt + dm1_opt * (x_data - x01_opt)) / (1 + np.exp(-k1_opt * (x_data - x01_opt)))
comp2 = (L2_opt + dm2_opt * (x_data - x02_opt)) / (1 + np.exp(-k2_opt * (x_data - x02_opt)))

# プロット: ベースライン
plt.plot(x_data, base, ':', color='black', label='Baseline (Background)')

# プロット: ベースラインの上に重なる各コンポーネントの寄与
plt.plot(x_data, base + comp1, '--', color='dodgerblue', alpha=0.8, label='Base + Process 1')
plt.plot(x_data, base + comp2, '--', color='limegreen', alpha=0.8, label='Base + Process 2')

# 変曲点を示す縦線
plt.axvline(x=x01_opt, color='dodgerblue', linestyle=':', alpha=0.6)
plt.axvline(x=x02_opt, color='limegreen', linestyle=':', alpha=0.6)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Additive Superimposition of Two Independent Sigmoid Processes')
plt.grid(True)
plt.tight_layout()
plt.show()