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

