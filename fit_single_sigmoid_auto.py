import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def exit_with_pause():
    input("\nEnterキーを押して終了してください...")
    sys.exit()

# 数理モデルの定義（単一シグモイド＋傾き変化）
def single_sigmoid_additive(x, C, m0, L, dm, k, x0):
    base = m0 * x + C
    comp = (L + dm * (x - x0)) / (1 + np.exp(-k * (x - x0)))
    return base + comp

# データの読み込みとパスの設定
if len(sys.argv) > 1:
    csv_file_path = sys.argv[1]
else:
    print("エラー：対象のCSVファイルをこのスクリプトのバッチファイルにドラッグ＆ドロップしてください。")
    exit_with_pause()

input_path = Path(csv_file_path)
output_dir = input_path.parent
base_name = input_path.stem

if input_path.suffix.lower() != '.csv':
    print(f"エラー：ドロップされたファイル '{input_path.name}' はCSVではありません。")
    exit_with_pause()

try:
    df = pd.read_csv(input_path)
except Exception as e:
    print(f"エラー：CSVファイルの読み込みに失敗しました。\n詳細：{e}")
    exit_with_pause()

if len(df.columns) < 2:
    print("エラー：CSVファイルには少なくとも2列のデータが必要です。")
    exit_with_pause()

col_x = df.columns[0]
col_y = df.columns[1]

df = df.dropna(subset=[col_x, col_y]).sort_values(by=col_x)
x_data = df.iloc[:, 0].values
y_data = df.iloc[:, 1].values

# 3. フィッティングの実行
n = len(x_data)
if n < 10:
    print("エラー：データ数が少なすぎます。")
    exit_with_pause()

# 初期と最終の20%のデータからベースラインを推論
m_start, C_start = np.polyfit(x_data[:int(n*0.2)], y_data[:int(n*0.2)], 1)
m_end, C_end     = np.polyfit(x_data[-int(n*0.2):], y_data[-int(n*0.2):], 1)

total_amplitude = np.mean(y_data[-int(n*0.2):]) - np.mean(y_data[:int(n*0.2)])
guess_x0 = np.percentile(x_data, 50)
guess_dm = m_end - m_start

initial_guess = [
    C_start, m_start,
    total_amplitude, guess_dm, 0.2, guess_x0
]

x_min, x_max = np.min(x_data), np.max(x_data)
# Bounds: L >= 0, k > 0, x0 はデータ範囲内
lower_bounds = [-np.inf, -np.inf, 0,      -np.inf, 0.001,   x_min]
upper_bounds = [ np.inf,  np.inf, np.inf,  np.inf, np.inf,  x_max]

try:
    popt, pcov = curve_fit(
        single_sigmoid_additive, 
        x_data, y_data, 
        p0=initial_guess, 
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )
except RuntimeError as e:
    print(f"\n最適化に失敗しました。データ形状が自動推論の限界を超えている可能性があります。\n詳細：{e}")
    exit_with_pause()
except Exception as e:
    print(f"\n予期せぬエラーが発生しました。\n詳細：{e}")
    exit_with_pause()

# 4. 結果の出力とテキストファイル保存
C_opt, m0_opt, L_opt, dm_opt, k_opt, x0_opt = popt

# 最適化されたパラメータを使って、Xに対するYの予測値を計算
y_fit = single_sigmoid_additive(x_data, *popt)
# 残差平方和と全変動を計算
ss_res = np.sum((y_data - y_fit) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
# R^2値の算出
r_squared = 1 - (ss_res / ss_tot)

result_text = (
    "最適化されたパラメータ (Single Sigmoid):\n"
    f"決定係数　　： {r_squared:.4f}\n"
    f"ベースライン： Y切片 C = {C_opt:.2f}, 初期傾き m0 = {m0_opt:.4f}\n"
    f"プロセス　　： 振幅 L = {L_opt:.2f}, 傾き変化量 dm = {dm_opt:.4f}, 急峻さ k = {k_opt:.4f}, 変曲点 x0 = {x0_opt:.2f}\n"
)
print("\n" + result_text)

txt_filename = f"{base_name}_fit_results_single.txt"
txt_output_path = output_dir / txt_filename

with open(txt_output_path, 'w', encoding='utf-8') as f:
    f.write(result_text)
print(f"※ パラメータを '{txt_output_path}' に保存しました。")

# 5. 結果のプロットと画像ファイル保存
plt.figure(figsize=(10, 7))
plt.scatter(x_data, y_data, label='Observed Data', color='gray', alpha=0.5, s=15)
plt.plot(x_data, single_sigmoid_additive(x_data, *popt), label='Fitted Model ($R^2={r_squared:.4f}$)', color='red', linewidth=2)

base = m0_opt * x_data + C_opt
comp = (L_opt + dm_opt * (x_data - x0_opt)) / (1 + np.exp(-k_opt * (x_data - x0_opt)))

plt.plot(x_data, base, ':', color='black', label='Baseline (Initial)')
plt.plot(x_data, base + comp, '--', color='dodgerblue', alpha=0.8, label='Process (Base + Sigmoid)')
plt.axvline(x=x0_opt, color='dodgerblue', linestyle=':', alpha=0.6)

plt.xlabel(col_x)
plt.ylabel(col_y)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'Fitting Single Sigmoid: {base_name}')
plt.grid(True)
plt.tight_layout()

png_filename = f"{base_name}_fitted_curve_single.png"
png_output_path = output_dir / png_filename

plt.savefig(png_output_path, dpi=300, bbox_inches='tight')
print(f"※ プロット画像を '{png_output_path}' に保存しました。")

plt.show()
print("\nすべての処理が完了しました。")
exit_with_pause()