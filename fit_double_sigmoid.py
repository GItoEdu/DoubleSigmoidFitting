import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 数理モデルの定義（完全な重ね合わせモデル）
def double_sigmoid_additive(x, C, m0, L1, dm1, k1, x01, L2, dm2, k2, x02):
    base = m0 * x + C
    comp1 = (L1 + dm1 * (x - x01)) / (1 + np.exp(-k1 * (x - x01)))
    comp2 = (L2 + dm2 * (x - x02)) / (1 + np.exp(-k2 * (x - x02)))
    return base + comp1 + comp2

# 2. データの読み込み
# 以下の変数を実際のCSVファイルに合わせて調整してください
csv_file_path = 'H:\\pH5_1.csv'
x_column_name = 'Temperature'
y_column_name = 'pH5_1'

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"エラー: '{csv_file_path}' が見つかりません。ファイル名とパスを確認してください。")
    exit()

# 欠損値の除外と、X軸に基づくソート（配列の前方/後方をスライスする推定ロジックに必須）
df = df.dropna(subset=[x_column_name, y_column_name])
df = df.sort_values(by=x_column_name)

x_data = df[x_column_name].values
y_data = df[y_column_name].values

# 3. フィッティングの実行（近接モデル向けのアプローチ）
n = len(x_data)
if n < 10:
    print("エラー: データ数が少なすぎます。")
    exit()

# 初期・最終ベースラインの概算
m_start, C_start = np.polyfit(x_data[:int(n*0.1)], y_data[:int(n*0.1)], 1)
m_end, C_end     = np.polyfit(x_data[-int(n*0.1):], y_data[-int(n*0.1):], 1)

# 全体のステップ幅(振幅)を算出
total_amplitude = np.mean(y_data[-int(n*0.1):]) - np.mean(y_data[:int(n*0.1)])

# 初期値の推論
# 実データに合わせて、x01とx02の初期値（変曲点のおおよそのX座標）を調整してください
guess_x01 = np.percentile(x_data, 40) # 例: 全体の中の40%位置
guess_x02 = np.percentile(x_data, 60) # 例: 全体の中の60%位置

initial_guess = [
    C_start, m_start,
    total_amplitude * 0.5, 0.0, 0.2, guess_x01,
    total_amplitude * 0.5, 0.0, 0.2, guess_x02
]

# 境界条件(Bounds)の設定
# X軸のスケールが0〜100ではない場合、x01, x02の上限(np.max(x_data))などを調整します
x_min, x_max = np.min(x_data), np.max(x_data)
lower_bounds = [-np.inf, -np.inf, 0,      -np.inf, 0.001,   x_min, 0,      -np.inf, 0.001,   x_min]
upper_bounds = [ np.inf,  np.inf, np.inf,  np.inf, np.inf,  x_max, np.inf,  np.inf, np.inf,  x_max]

try:
    successful_popt = [
        16086.68, -293.7046,
        11721.57, -43.9145, 0.6193, 33.11,
        32463.22, -330.4160, 0.1258, 50.47
    ]
    popt, pcov = curve_fit(
        double_sigmoid_additive, 
        x_data, y_data, 
        p0=successful_popt, 
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )
except RuntimeError as e:
    print(f"最適化に失敗しました。初期値やBoundsを見直してください。\n詳細: {e}")
    exit()

# 4. 結果の出力
C_opt, m0_opt, L1_opt, dm1_opt, k1_opt, x01_opt, L2_opt, dm2_opt, k2_opt, x02_opt = popt

print("最適化されたパラメータ:")
print(f"ベースライン: Y切片 C = {C_opt:.2f}, 初期傾き m0 = {m0_opt:.4f}")
print(f"プロセス1   : 振幅 L1 = {L1_opt:.2f}, 傾き変化量 dm1 = {dm1_opt:.4f}, 急峻さ k1 = {k1_opt:.4f}, 変曲点 x01 = {x01_opt:.2f}")
print(f"プロセス2   : 振幅 L2 = {L2_opt:.2f}, 傾き変化量 dm2 = {dm2_opt:.4f}, 急峻さ k2 = {k2_opt:.4f}, 変曲点 x02 = {x02_opt:.2f}")

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

plt.xlabel(x_column_name)
plt.ylabel(y_column_name)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Fitting Double Sigmoid to CSV Data')
plt.grid(True)
plt.tight_layout()
plt.show()