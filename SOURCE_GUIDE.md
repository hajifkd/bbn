# ソースコードガイド

この文書は、現在の公開計算経路を中心に各ソースの役割を簡単に説明する。
物理・数値計画の詳細は`FOKKER_PLANCK_IMPLEMENTATION_PLAN.md`、進捗は
`FOKKER_PLANCK_PROGRESS.md`を参照すること。

## 計算全体の流れ

```text
CascadeConfig + MonochromaticPhotonSource
                  │
                  ▼
            CascadeSolver
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
IC以外のBoltzmann項      逆Compton散乱
hybrid_collision.py      inverse_compton_split.py
       │                     │
       └──────────┬──────────┘
                  ▼
     光子・電子の疎な定常連立方程式
       + 電子のChang–Cooper FP flux
                  │
                  ▼
       CascadeSpectrum（CSV・診断値）
```

エネルギーはすべてGeV、時間・密度を含む量は自然単位系で扱う。

## 公開APIと入力

### `src/bbn_cascade/__init__.py`

利用者向けの入口。現在は次を公開する。

- `CascadeConfig`
- `CascadeSolver`
- `CascadeSpectrum`
- `MonochromaticPhotonSource`

ここからimportされる`CascadeSolver`は`hybrid_solver.py`の実装である。

### `src/bbn_cascade/config.py`

温度、最大・最小エネルギー、格子数、背景バリオン密度、求積次数、ICの
soft/hard分割値、diffusionの有無などを保持し、入力値を検証する。

主要設定は`soft_fraction=0.05`、`quadrature_order`、
`background_cutoff_over_temperature=200`など。

### `src/bbn_cascade/sources.py`

単色光子sourceを定義する。既定では総エネルギー注入率を1とし、光子数
注入率を`energy_injection_rate / energy`として返す。

### `src/bbn_cascade/constants.py`

微細構造定数、電子質量、古典電子半径、GeVへの単位換算、熱浴光子数密度を
定義する。

## 物理過程

### `src/bbn_cascade/processes.py`

エネルギー再分配を積分した全断面積・loss rateの基礎関数を置く。

- Klein–Nishina Compton全断面積
- Maximonの原子核場対生成全断面積
- 光子–光子弾性散乱rate
- 背景電子・原子核の密度

一部の簡易rateは比較用コードでも使われる。hybrid本計算では、必要に応じて
`hybrid_collision.py`が完全積分rateを組み立てる。

### `src/bbn_cascade/kernels.py`

astro-ph/9412055 Appendixに対応する微分kernelと熱浴積分を実装する。

- Jones逆Compton関数
- γγ対生成の電子分布関数
- Bethe–Heitler原子核場対生成kernel
- Compton散乱kernel
- Planck分布上で積分したγγ対生成rate

4種類の遷移は、出射種・入射種を光子`γ`と電子・陽電子`e`として
`K_γγ`, `K_γe`, `K_eγ`, `K_ee`に対応する。

## Hybrid Boltzmann–Fokker–Planck計算

### `src/bbn_cascade/inverse_compton_split.py`

逆Compton散乱専用の共通積分器。電子の相対エネルギー損失
`u = ΔE/(E-m_e)`でsoft/hard領域を分け、同じJones kernel積分から次を作る。

- soft drift `b(E)`
- soft diffusion `D(E)`
- hard loss rate
- soft/hard光子再注入kernel
- hard電子再注入kernel

再注入kernelは、散乱後エネルギーの求積点を単一binへ投票せず、各娘粒子
セルの境界で積分区間を分割してセル内積分する。hard領域では電子側の
セル境界も光子エネルギーへ写し、光子・電子を同じ積分から再注入する。

### `src/bbn_cascade/inverse_compton_fp.py`

ICモーメントを保持する`InverseComptonMoments`と、有限体積fluxに使う
Chang–Cooper重みを定義する。古い共通積分器も残っているが、現在の本計算は
`inverse_compton_split.py`の積分器を使用する。

### `src/bbn_cascade/hybrid_collision.py`

逆Compton以外の光子親Boltzmann項を娘粒子セル上で積分する。

- 光子–光子弾性散乱による`γ -> γ`
- 背景電子Compton散乱による`γ -> γ,e`
- 原子核場対生成による`γ -> e`
- 背景光子とのγγ対生成による`γ -> e`

論文Eq. (31)の光子–光子散乱近似は、Eq. (32)に従って
`E <= m_e^2/T`だけに適用する。lossと再注入kernelの両方に同じcutoffを使う。

### `src/bbn_cascade/hybrid_solver.py`


本計算の中心。対数エネルギー格子を作り、上記2系統のcollision項を結合して
光子・電子の定常方程式を疎行列として解く。

電子soft IC項はChang–Cooper有限体積flux、hard ICとその他の反応は
Boltzmann遷移行列として入る。低エネルギー端はoutflow、高エネルギー端は
上側流入なし。解に許容値を超える負binがあれば例外にする。

結果は`CascadeSpectrum`として返し、`to_csv()`でエネルギー、光子・電子
スペクトル、各loss rateを保存できる。

## 比較・CLI

### `src/bbn_cascade/published.py`

astro-ph/9412055 Table 1の`P_low`, `N_low`, `P_pp`, `N_pp`を保持し、論文の
power-law fitを計算する。完全な数値計算ではなく比較表示専用。

### `src/bbn_cascade/cli.py`

- `bbn-cascade solve`: 1条件の定常スペクトルをCSVとPNGへ保存
- `bbn-cascade reproduce-9412055`: Fig.1–3相当の6条件を計算

後者はhybrid-FP数値解を実線、Table 1 fitを破線で描き、各数値解もCSVへ
保存する。

## 旧・実験的コード

`solver.py`, `exact_solver.py`, `finite_solver.py`, `finite_volume.py`は開発途中の
別方式であり、現在の公開APIやCLIからは使われない。内容が現行修正に追随して
いない可能性があるため、物理結果の生成には使用しないこと。

`config.py.rej`や`.orig`ファイルも失敗したパッチ・バックアップの残骸であり、
実行時には参照されない。将来の整理対象である。

## テスト

`tests/`には断面積、微分kernel、Jones関数のsupport、FPモーメント、
Chang–Cooper極限、Eq. (32) cutoff、定常solver、CLI単位変換のテストがある。

```bash
uv run pytest
```

現状のテスト数は11。粒子数・エネルギー収支、格子収束、完全Boltzmann比較は
今後追加する予定である。

## 最初に読む順序

実際の計算を追う場合は、次の順序が分かりやすい。

1. `config.py`と`sources.py`で入力を確認する。
2. `hybrid_solver.py`の`CascadeSolver.solve()`から全体構造を見る。
3. `hybrid_collision.py`で光子親のBoltzmann項を見る。
4. `inverse_compton_split.py`でICのsoft/hard分割を見る。
5. `kernels.py`と`processes.py`で個々の物理式を確認する。
6. `cli.py`でFig.1–3の条件と出力方法を確認する。

## 熱浴と出射端の複合求積（2026-09-17更新）

`quadrature.py` は `epsilon/T` の熱分布を小区間に分けたGauss求積を提供する。
`kernels.py` の対生成の全率・微分kernelと、公開経路の
`inverse_compton_split.py` が使用する。ICではsoft/hard分割が開始する
熱浴エネルギーも区間境界に加える。

`hybrid_collision.py` の `photon_parent_column()` は、出射格子だけでなく
`E_in - E_out` に写した格子でも積分区間を分ける。電子の高エネルギー端の
細い分布を、カスケード格子全体を増やさずに積分するための処理である。

`tests/test_composite_quadrature.py` でPlanck分布の粒子数・エネルギーモーメント、
対生成の粒子数・エネルギー収支、ICの求積次数依存を確認する。
`reproduce-9412055` はスペクトルと図に加え、各条件のJSONとセル境界NPYを保存する。

### FPフラックスのD微分項

`inverse_compton_fp.py` の `chang_cooper_flux_coefficients()` が
`J=-(b+D'/2)f-D*f'/2` のface係数を計算する。公開solverはこの関数を使う。
`tests/test_fp_flux.py` にDのエネルギー依存を含む解析的検証がある。
`compare_fp_flux_grids.py` は修正後の120/240点/decadeを比較する。
