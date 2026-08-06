# Hybrid Fokker–Planck実装の進捗

更新日: 2026-08-06

対象計画: [`FOKKER_PLANCK_IMPLEMENTATION_PLAN.md`](FOKKER_PLANCK_IMPLEMENTATION_PLAN.md)

## 現在の到達点

astro-ph/9412055の電磁カスケードについて、逆Compton散乱をsoft領域の
Fokker–Planck項とhard領域の完全Boltzmann kernelに分けた定常ソルバーを
公開APIとCLIの既定計算経路にした。Fig.1–3に対応する6条件を計算し、
スペクトル画像とCSVを生成できる。

現時点では、物理過程と主要な数値解法の実装は完了している。一方、計画で
要求した全収支診断、完全Boltzmann解との比較、格子収束の自動化、Table 1の
10%再現は未完了である。

## 計画項目ごとの状態

| 項目 | 状態 | 内容 |
|---|---|---|
| IC soft/hard分割 | 完了 | `u = ΔE/(E-m_e)`、既定値`u_split=0.05` |
| drift `b(E)` | 完了 | Jones微分率から数値積分 |
| diffusion `D(E)` | 完了 | 同じ積分点から二次モーメントを計算 |
| hard IC loss | 完了 | soft領域との二重計上なし |
| soft/hard光子kernel | 完了 | soft光子も捨てず光子方程式へ再注入 |
| hard電子kernel | 完了 | hard散乱後の電子を完全kernelで再注入 |
| セル内kernel積分 | 完了 | 求積点の単一bin投票を廃止し、出力セルごとに直接積分 |
| finite volume | 完了 | 電子FP fluxを保存形で離散化 |
| Chang–Cooper | 完了 | driftとdiffusionをcell faceで結合 |
| 光子・電子連立解 | 完了 | sparseな大域定常方程式として同時に解く |
| 負値処理 | 完了 | 有意な負binでは失敗し、丸め誤差だけゼロ化 |
| 高エネルギー境界 | 完了 | 上側流入なし、単色光子sourceを最上位cellへ注入 |
| 低エネルギーoutflow | 部分完了 | outflow条件は実装済み、thermal deposition記録は未実装 |
| 設定API | 部分完了 | 計画の設定値は追加済み。ただし`integration_rtol`と`boltzmann`切替は本計算で未活用 |
| 診断 | 部分完了 | 負bin、softエネルギー整合性、drift/diffusion差を出力 |
| Fig.1–3 | 完了 | 10 TeV、1 TeV、100 GeV × 1 eV、100 eVを生成 |
| Table 1の10%再現 | 未完了 | 傾向と指数は概ね再現するが、全係数の10%一致は未達 |

## 実装ファイル

- `src/bbn_cascade/hybrid_solver.py`: 光子・電子の定常連立ソルバー
- `src/bbn_cascade/inverse_compton_split.py`: IC soft/hard分割、FPモーメント、セル積分kernel
- `src/bbn_cascade/inverse_compton_fp.py`: データ型とChang–Cooper重み
- `src/bbn_cascade/hybrid_collision.py`: IC以外のBoltzmann loss/kernel
- `src/bbn_cascade/kernels.py`: Jones、γγ対生成、Bethe–Heitler、Compton微分kernel
- `src/bbn_cascade/processes.py`: 全断面積と背景密度
- `src/bbn_cascade/cli.py`: 単一計算とFig.1–3再現CLI

公開APIの`CascadeSolver`は`hybrid_solver.py`を使用する。過去の実験的solver
ファイルは残っているが、公開計算経路では使用しない。

## 今回までに修正した重要事項

1. γγ対生成の背景光子thresholdを`m_e^2/E_gamma`へ修正。
2. Jones関数の分母とCompton微分kernelの符号を修正。
3. γγ対生成電子kernelへ正しい運動学的supportを追加。
4. 光子–光子弾性散乱の係数とPlanck momentを修正。
5. Maximon核対生成全断面積を論文Eq. (34)、(36)へ合わせた。
6. 論文Eq. (31)の光子–光子散乱近似を`E <= m_e^2/T`に限定し、
   Eq. (32)どおり高エネルギー側のlossと再注入kernelをゼロ化。
7. IC再注入を求積点のhistogram投票から娘粒子セル内の直接積分へ変更。
   hard電子側のセル境界も同時に積分区間へ入れている。

Eq. (32)のcutoff漏れは高温・高エネルギーほど重大だった。修正前の
`T=100 eV, E0=10 TeV, E_gamma=0.1 E0`では、適用範囲外の光子–光子
散乱rateがγγ対生成rateの約`2.8e4`倍になっていた。修正により同領域の
スペクトルは約4.35 dex上昇した。

## 現在の検証結果

- `uv run pytest`: 11 tests passed
- Jones kernelのsupportと非負性: 合格
- 4種類の微分kernelの有限性・非負性: 合格
- `b(E)`, `D(E)`, hard lossの非負性: 合格
- soft光子エネルギーモーメントと`b(E)`の同一積分上の一致: 合格
- Chang–Cooperのdrift極限: 合格
- hybrid定常解の非負性: 合格
- 光子–光子散乱の論文Eq. (32) cutoff: 回帰テスト追加済み

手動検証では`soft_fraction=0.02, 0.05, 0.1`による主要なスペクトル指数の
変化は小さく、FP diffusionの影響も通常は数%以下だった。ただし計画で
要求した「主要領域で2%以内」を全6条件について自動判定するテストは未実装。

## 生成物と再現方法

出力先:

```text
results/9412055_hybrid/
├── figure_1.png  # E0 = 10 TeV
├── figure_2.png  # E0 = 1 TeV
├── figure_3.png  # E0 = 100 GeV
├── E0_*GeV_T_*eV.csv
└── table1_comparison.csv
```

再計算:

```bash
uv run bbn-cascade reproduce-9412055 \
  --points-per-decade 60 \
  --quadrature-order 8 \
  --soft-fraction 0.05 \
  --output results/9412055_hybrid
```

図ではhybrid-FP数値解を実線、Table 1のpower-law fitを破線で表示する。
縦軸範囲は`log10(f_gamma/GeV^2) = [10, 50]`。破線の高エネルギーcutoffは
比較表示用であり、Table 1 fitの本来の適用領域外では定量比較に使わない。

## 未完了項目と次の優先順位

1. 定常方程式のcellごとの残差、粒子数収支、エネルギー収支を実装する。
2. 低エネルギー境界を通るthermal depositionを記録する。
3. soft+hard IC rateと分割前の完全rateの一致を自動テストする。
4. 一定`b`・単色電子sourceのdrift-only解析解テストを追加する。
5. `soft_fraction=0.02/0.05/0.1`の6条件比較を自動化する。
6. 60/120/240点/decadeの格子収束を自動化する。
7. 完全Boltzmann IC解を比較用経路として完成させ、`u_split -> 0`を検証する。
8. Fig.1–3を定量比較し、Table 1の指数と規格化を10%以内へ詰める。
9. 未使用の旧solverと`.orig`/`.rej`ファイルを整理し、READMEを現状へ更新する。

## 現時点の評価

hybrid Fokker–Planck法の中心部分とFig.1–3の生成経路は動作している。
