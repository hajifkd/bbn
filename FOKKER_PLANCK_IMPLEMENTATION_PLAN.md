# 逆Compton散乱のHybrid Fokker–Planck実装方針

## 目的

astro-ph/9412055 の定常電磁カスケードを、逆Compton散乱の
`E_out ≈ E_in` に集中する電子kernelで生じる悪条件性を避けて解く。

小さな電子エネルギー移行はFokker–Planck項、大きな移行は完全な
微分Boltzmann kernelとして扱う。光子生成kernelは全領域で保持し、
電子が失ったエネルギーと生成光子のエネルギーを同じ積分から計算する。

## 1. 物理的な分割

逆Compton散乱

\[
e(E)+\gamma_{\rm BG}(\epsilon)
\rightarrow e(E-\Delta E)+\gamma(E_\gamma),
\qquad \Delta E=E_\gamma-\epsilon
\]

を、電子の相対エネルギー損失

\[
u\equiv\frac{\Delta E}{E-m_e}
\]

で分割する。

- soft領域: \(0<u<u_{\rm split}\)
- hard領域: \(u\ge u_{\rm split}\)

既定値は `u_split = 0.05` とする。`0.02`, `0.05`, `0.1` で結果が収束することを必須検証とし、特定の格子幅を分割基準にしない。

二重計上を避けるため、soft領域は電子の積分kernelから完全に除外し、Fokker–Planck係数だけへ入れる。hard領域だけを \(K_{ee}^{\rm hard}\) と \(\Gamma_e^{\rm hard}\) に使う。

## 2. Fokker–Planck係数

Jonesの完全微分率を \(d^2\Gamma_{\rm IC}/(d\epsilon\,dE_\gamma)\) とし、背景Planck分布を含めて次を数値積分する。

\[
b(E)=\int_0^\infty d\epsilon
\int_{\epsilon}^{E_{\gamma,\max}}dE_\gamma\,
\Delta E\,
\frac{d^2\Gamma_{\rm IC}}{d\epsilon\,dE_\gamma}
\Theta(u_{\rm split}-u),
\]

\[
D(E)=\int_0^\infty d\epsilon
\int_{\epsilon}^{E_{\gamma,\max}}dE_\gamma\,
(\Delta E)^2
\frac{d^2\Gamma_{\rm IC}}{d\epsilon\,dE_\gamma}
\Theta(u_{\rm split}-u).
\]

ここで

\[
E_{\gamma,\max}
=\frac{4\epsilon E^2}{m_e^2+4\epsilon E},
\qquad
\Delta E=E_\gamma-\epsilon.
\]

背景光子積分は `epsilon/T` の対数変数で行い、既定上限を `200` とする。積分許容誤差は設定可能にし、既定相対誤差を `1e-4` とする。

電子のsoft衝突項は

\[
C_{ee}^{\rm soft}[f_e]
=\frac{\partial}{\partial E}[b(E)f_e(E)]
+\frac12\frac{\partial^2}{\partial E^2}[D(E)f_e(E)]
\]

とする。最初にdriftのみ、次にdiffusion込みを実装し、diffusionを無効化した結果との差も診断へ出す。

## 3. 光子生成との整合性

soft散乱でも光子は生成されるため、\(K_{\gamma e}\) は捨てない。

- `K_gamma_e_soft`: \(u<u_{\rm split}\) のJones kernel
- `K_gamma_e_hard`: \(u\ge u_{\rm split}\) のJones kernel
- 実際の光子方程式には三者の和を入れる

各親電子エネルギーで

\[
b(E)=\int dE_\gamma\,(E_\gamma-\epsilon)
K_{\gamma e}^{\rm soft}
\]

が同じ数値積分から得られるようにし、独立に計算した近い数の差を使わない。

## 4. 定常方程式

電子方程式は

\[
0=
\frac{\partial}{\partial E}(bf_e)
+\frac12\frac{\partial^2}{\partial E^2}(Df_e)
-\Gamma_e^{\rm hard}f_e
+\int_E^{E_{\max}}dE'\,
K_{ee}^{\rm hard}(E,E')f_e(E')
+\int_E^{E_{\max}}dE'\,
K_{e\gamma}(E,E')f_\gamma(E')
+S_e(E).
\]

光子方程式は完全な積分kernelのままとする。

\[
0=-\Gamma_\gamma f_\gamma
+\int K_{\gamma\gamma}f_\gamma
+\int(K_{\gamma e}^{\rm soft}+K_{\gamma e}^{\rm hard})f_e
+S_\gamma.
\]

電子と光子を各エネルギーbinで同時に解き、高エネルギー側から低エネルギー側へ進む。

## 5. 離散化

Fokker–Planck部分は粒子数保存型finite volumeで離散化する。

\[
0=-\frac{J_{i+1/2}-J_{i-1/2}}{\Delta E_i}
+C_i^{\rm hard}+S_i,
\]

\[
J=-bf_e-\frac12\frac{\partial(Df_e)}{\partial E}.
\]

- drift fluxは冷却方向に対するupwindを使う
- diffusion fluxはcell faceで中心差分する
- Chang–Cooper重みを実装し、非負性と定常解の安定性を確保する
- kernelは点値ではなく娘粒子bin上で積分する
- 同一binのhard遷移は局所2成分行列へ入れる
- 解の負値を黙ってfloorしない。負値が許容値を超えたら計算を失敗させる

## 6. 境界条件

- 高エネルギー端: 上側からの流入なし。単色sourceは最上位cellのsourceとして入れる
- 低エネルギー端: outflow境界。下端を通過した粒子数・エネルギーをthermal depositionとして記録する
- diffusion項には高エネルギー端でzero diffusive flux、低エネルギー端でoutflowを課す

## 7. API変更

`CascadeConfig`へ以下を追加する。

- `inverse_compton_scheme: Literal["boltzmann", "hybrid-fp"] = "hybrid-fp"`
- `soft_fraction: float = 0.05`
- `include_energy_diffusion: bool = True`
- `integration_rtol: float = 1e-4`
- `background_cutoff_over_T: float = 200.0`

新しい内部データ型 `InverseComptonMoments` はエネルギー格子上の `drift`, `diffusion`, `hard_loss`, `soft_photon_kernel`, `hard_photon_kernel`, `hard_electron_kernel` を保持する。

出力診断へ以下を追加する。

- 粒子数収支
- エネルギー収支
- 最小スペクトル値と負bin数
- `soft_fraction` 変更による最大相対差
- driftのみとdrift+diffusionの最大相対差
- 各親電子binでのsoft光子エネルギーと \(b(E)\) の整合性

## 8. テスト

1. Jones kernelの運動学的範囲外がゼロになる。
2. `b(E) >= 0`, `D(E) >= 0`, `hard_loss >= 0`。
3. soft/hardの和が分割前の全逆Compton rateと積分誤差内で一致する。
4. soft光子kernelのエネルギーモーメントが `b(E)` と一致する。
5. drift-onlyで単色電子注入・一定 `b` の解析解 \(f=\dot n/b\) を再現する。
6. finite-volume fluxがsourceを除いて電子数を保存する。
7. `soft_fraction = 0.02, 0.05, 0.1` の光子スペクトルが主要領域で2%以内に収束する。
8. 60, 120, 240点/decadeで主要領域が収束し、負binを生じない。
9. astro-ph/9412055 Fig.1–3の6条件を完全計算から生成する。
10. Table 1の \(P_{\rm low},P_{\rm pp},N_{\rm low},N_{\rm pp}\) を10%以内で再現する。

## 9. 実装順序

1. 現在の代表エネルギー近似・positivity limiterを本計算経路から外す。
2. Jones kernelをsoft/hardに分割する共通積分器を実装する。
3. `b`, `D`, hard loss、光子・電子kernelを同じ積分から生成する。
4. drift-only finite-volume定常ソルバーを解析解で検証する。
5. Chang–Cooper diffusionを追加する。
6. 光子方程式および他の完全kernelと結合する。
7. 分割依存・格子収束・収支を自動検証する。
8. Fig.1–3とTable 1を再計算し、`published-fit` は比較用にのみ残す。

## 参考上の位置づけ

このhybrid法はastro-ph/9412055およびACROPOLISの点格子・積分kernel解法そのものではない。両者と同じ微分反応率を使いつつ、逆Comptonの対角近傍をKramers–Moyal展開して数値的に安定化する実装である。結果の正当性は `soft_fraction -> 0`、格子収束、完全Boltzmann解との比較で確認する。
