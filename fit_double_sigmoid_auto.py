import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import japanize_matplotlib
plt.rcParams['font.family'] = "Gen Jyuu Gothic LP"

def exit_with_pause():
    input("\nEnterキーを押して終了してください...")
    sys.exit()

# 数理モデルの定義
def double_sigmoid_additive(x, C, m0, L1, dm1, k1, x01, L2, dm2, k2, x02):
    base = m0 * x + C
    comp1 = (L1 + dm1 * (x - x01)) / (1 + np.exp(-k1 * (x - x01)))
    comp2 = (L2 + dm2 * (x - x02)) / (1 + np.exp(-k2 * (x - x02)))
    return base + comp1 + comp2

# 入力ファイルパスを取得
if len(sys.argv) > 1:
    csv_file_path = sys.argv[1]
else:
    print("エラー：対象のCSVファイルをドラッグ＆ドロップしてください。")
    exit_with_pause()

input_path = Path(csv_file_path)

# データ出力先
output_dir = input_path.parent # フォルダ名を取得
base_name = input_path.stem # 拡張子を除いたファイル名を取得

try:
    df = pd.read_csv(input_path)
except Exception as e:
    print(f"エラー: CSVファイルの読み込みに失敗しました。\n詳細： {e}")
    exit_with_pause()

if len(df.columns) < 2:
    print("エラー：CSVファイルには少なくとも2列のデータが必要です。")
    exit_with_pause()

col_x = df.columns[0]
col_y = df.columns[1]

df = df.dropna(subset=[col_x, col_y]).sort_values(by=col_x)

x_data = df.iloc[:, 0].values
y_data = df.iloc[:, 1].values

# フィッティングの実行 (自動推論ロジック)
n = len(x_data)
if n < 10:
    print("エラー: データ数が少なすぎます。")
    exit_with_pause()

m_start, C_start = np.polyfit(x_data[:int(n*0.1)], y_data[:int(n*0.1)], 1)
total_amplitude = np.mean(y_data[-int(n*0.1):]) - np.mean(y_data[:int(n*0.1)])

guess_x01 = np.percentile(x_data, 40)
guess_x02 = np.percentile(x_data, 60)

initial_guess = [
    C_start, m_start,
    total_amplitude * 0.5, 0.0, 0.2, guess_x01,
    total_amplitude * 0.5, 0.0, 0.2, guess_x02
]

x_min, x_max = np.min(x_data), np.max(x_data)
lower_bounds = [-np.inf, -np.inf, 0,      -np.inf, 0.001,   x_min, 0,      -np.inf, 0.001,   x_min]
upper_bounds = [ np.inf,  np.inf, np.inf,  np.inf, np.inf,  x_max, np.inf,  np.inf, np.inf,  x_max]

try:
    popt, pcov = curve_fit(
        double_sigmoid_additive, 
        x_data, y_data, 
        p0=initial_guess, 
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )
except RuntimeError as e:
    print(f"最適化に失敗しました。\n詳細：{e}")
    exit_with_pause()

# 結果の出力とテキストファイル保存
C_opt, m0_opt, L1_opt, dm1_opt, k1_opt, x01_opt, L2_opt, dm2_opt, k2_opt, x02_opt = popt

# 最適化されたパラメータを使って、Xに対するYの予測値を計算
y_fit = double_sigmoid_additive(x_data, *popt)
# 残差平方和と全変動を計算
ss_res = np.sum((y_data - y_fit) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
# R^2値の算出
r_squared = 1 - (ss_res / ss_tot)

result_text = (
    "最適化されたパラメータ (Auto):\n"
    f"決定係数　　： {r_squared:.4f}\n"
    f"ベースライン： Y切片 C = {C_opt:.2f}, 初期傾き m0 = {m0_opt:.4f}\n"
    f"プロセス１　： 振幅 L1 = {L1_opt:.2f}, 傾き変化量 dm1 = {dm1_opt:.4f}, 急峻さ k1 = {k1_opt:.4f}, 変曲点 x01 = {x01_opt:.2f}\n"
    f"プロセス２　： 振幅 L2 = {L2_opt:.2f}, 傾き変化量 dm2 = {dm2_opt:.4f}, 急峻さ k2 = {k2_opt:.4f}, 変曲点 x02 = {x02_opt:.2f}\n"
)
print(result_text)

# ファイル名の構築
txt_filename = f"{base_name}_fit_results_auto.txt"
txt_output_path = output_dir / txt_filename

with open(txt_output_path, 'w', encoding='utf-8-sig') as f:
    f.write(result_text)
print(f"※ パラメータを '{txt_output_path}' に保存しました。")

# 結果のプロットと画像ファイル保存
plt.figure(figsize=(10, 7))
plt.scatter(x_data, y_data, label='Observed Data', color='gray', alpha=0.5, s=15)
plt.plot(x_data, double_sigmoid_additive(x_data, *popt), label=f"Fitted Model ($R^2={r_squared:.4f}$)", color='red', linewidth=2)

base = m0_opt * x_data + C_opt
comp1 = (L1_opt + dm1_opt * (x_data - x01_opt)) / (1 + np.exp(-k1_opt * (x_data - x01_opt)))
comp2 = (L2_opt + dm2_opt * (x_data - x02_opt)) / (1 + np.exp(-k2_opt * (x_data - x02_opt)))

plt.plot(x_data, base, ':', color='black', label='Baseline')
plt.plot(x_data, base + comp1, '--', color='dodgerblue', alpha=0.8, label='Process 1')
plt.plot(x_data, base + comp2, '--', color='limegreen', alpha=0.8, label='Process 2')
plt.axvline(x=x01_opt, color='dodgerblue', linestyle=':', alpha=0.6)
plt.axvline(x=x02_opt, color='limegreen', linestyle=':', alpha=0.6)

plt.xlabel(col_x)
plt.ylabel(col_y)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

png_filename = f"{base_name}_fitted_curve_auto.png"
png_output_path = output_dir / png_filename
plt.savefig(png_output_path, dpi=300, bbox_inches='tight')
print(f"※ プロット画像を '{png_output_path}' に保存しました。")
print("\n全ての処理が完了しました。")
exit_with_pause()