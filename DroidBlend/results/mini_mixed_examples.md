# DroidBlend examples

Three datasets, two samples each, with four core experiment outputs grouped by sample.

## hotpotqa

### Sample 1
- ID: `5a84c4135542994c784dda31`
- Question: Are Yingkou and Fuding the same level of city?
- Gold: no

Prompt excerpt:

```text
Yingkou Yingkou () is a prefecture-level city of Liaoning province, People's Republic of China. It is a port city of the Bohai Sea, and is the location of the mouth of the Liao River. Fuding () is a county-level city in northeastern Ningde prefecture level city, on Fujian's border with Zhejiang province. Question: Are Yingkou and Fuding the same level of city? Answer:
```

Outputs:

- full_prefill (0.03s)
```text
yes
```
- full_kv_reuse (0.02s)
```text
Yingkou is a prefecture-level city of Liaoning province, China. Fuding is a county-level city of Fujian province, China.
```
- droidspeak (0.08s)
```text
no
```
- cacheblend_reference (0.08s)
```text

```

### Sample 2
- ID: `5ab642845542995eadeeff8e`
- Question: Are Broughtonia and Laeliocattleya both orchids?
- Gold: yes

Prompt excerpt:

```text
Laeliocattleya Laeliocattleya is a nothogenus of intergeneric orchid hybrids descended from the parental genera "Laelia" and "Cattleya". It is abbreviated Lc. in the horticultural trade. Broughtonia Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles. The genus is abbreviated Bro in trade journals. Question: Are Broughtonia and Laeliocattleya both orchids? Answer:
```

Outputs:

- full_prefill (0.03s)
```text
yes
```
- full_kv_reuse (0.02s)
```text
Laeliocattleya is a nothogenus of intergeneric orchid hybrids descended from the parental genera Laelia and Cattleya. The name is derived from the Greek word laelios, meaning "orchid", and the genus Cattley
```
- droidspeak (0.04s)
```text
Yes
```
- cacheblend_reference (0.05s)
```text
Yes
```

## 2wikimqa

### Sample 1
- ID: `9df72caf097511ebbdb0ac1f6bf848b6`
- Question: Which film came out earlier, Mi Viudo Y Yo or El Siete Machos?
- Gold: El Siete Machos

Prompt excerpt:

```text
El Siete Machos El Siete Machos is a 1951 Mexican western comedy film directed by Miguel M. Delgado, and starring Cantinflas, Alma Rosa Aguirre, and Miguel Ángel Ferriz. Mi viudo y yo Mi viudo y yo is a 1954 Argentine film. Question: Which film came out earlier, Mi Viudo Y Yo or El Siete Machos? Answer:
```

Outputs:

- full_prefill (0.02s)
```text
El Siete Machos
```
- full_kv_reuse (0.02s)
```text

```
- droidspeak (0.04s)
```text
El Siete Machos
```
- cacheblend_reference (0.05s)
```text

```

### Sample 2
- ID: `ba9b7cea0bdb11eba7f7acde48001122`
- Question: What is the date of birth of the director of film You'Re My Everything (Film)?
- Gold: August 10, 1896

Prompt excerpt:

```text
You're My Everything (film) You're My Everything is a 1949 film directed by Walter Lang and starring Dan Dailey and Anne Baxter. Walter Lang Walter Lang (August 10, 1896 – February 7, 1972) was an American film director. Question: What is the date of birth of the director of film You'Re My Everything (Film)? Answer:
```

Outputs:

- full_prefill (0.02s)
```text
August 10, 1896
```
- full_kv_reuse (0.02s)
```text
Walter Lang (June 2, 1904 – February 14, 1972) was an American film director. He was born in New York City, New York, U.S. and died in Los Angeles, California, U.S.
```
- droidspeak (0.04s)
```text
August 10, 1896
```
- cacheblend_reference (0.05s)
```text
1904-02-1972
```

## multifieldqa_en

### Sample 1
- ID: `22034e095a602824678c4028e6f605919ce520270dc06089`
- Question: What is the scaling form for the alternative order parameter O?
- Gold: O(t, L_{\parallel}; S_\Delta) = L_{\parallel}^{-\beta/[\nu(1+\Delta)]} \tilde f_O(t/L_{\parallel}^{z/(1+\Delta)}; S_\Delta).

Prompt excerpt:

```text
\section*{Dynamical Behaviour of $O$ in Lattice Gases} The dynamical behaviour of the anisotropic order parameter $m$ [see Eq.~\eqref{eq:def-m} in the Letter] following a quench to the critical point is well described by the Gaussian theory for all the three lattice gas models studied, $i.e.,$ driven lattice gas with either constant (IDLG) or random (RDLG) infinite drive and equilibrium lattice gas (LG). In other words, in the short-time regime, $m \sim t^{1/2}$ [see Eq. \eqref{eq:mt}] and the Binder cumulant $g$ of the lowest transverse mode [defined in Eq. \eqref{eq:binder}] is zero in this regime. The alternative order parameter $O,$ however, distinguishes between the driven (IDLG, RDLG) and the equilibrium (LG) lattice gases. In order to understand this, we first write the phenomenological scaling form for $O$, analogous to Eq. \eqref{eq:scalingass} in the Letter, \begin{eqnarray} O (t, L_{\parallel} ; S_\Delta) = L_{\parallel}^{-\beta/[\nu(1+\Delta)]} \tilde f_O (t/L_{\parallel...
```

Outputs:

- full_prefill (0.23s)
```text
$O \sim t^{\alpha} L_\parallel^{-1/2}$
```
- full_kv_reuse (0.02s)
```text
$\mu{figures the same exponent $naundayed BDLG models.
```
- droidspeak (0.10s)
```text
$O (t $t$ $ O (t$ $ $ $ $ O (t$ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
```
- cacheblend_reference (0.22s)
```text
(a) $n't for the LG model.
```

### Sample 2
- ID: `322b4b2b074049aac101086797d655a21671ef5a9f366353`
- Question: What is the dynamical behavior of the anisotropic order parameter following a quench to the critical point?
- Gold: It is well described by the Gaussian theory.

Prompt excerpt:

```text
\section*{Dynamical Behaviour of $O$ in Lattice Gases} The dynamical behaviour of the anisotropic order parameter $m$ [see Eq.~\eqref{eq:def-m} in the Letter] following a quench to the critical point is well described by the Gaussian theory for all the three lattice gas models studied, $i.e.,$ driven lattice gas with either constant (IDLG) or random (RDLG) infinite drive and equilibrium lattice gas (LG). In other words, in the short-time regime, $m \sim t^{1/2}$ [see Eq. \eqref{eq:mt}] and the Binder cumulant $g$ of the lowest transverse mode [defined in Eq. \eqref{eq:binder}] is zero in this regime. The alternative order parameter $O,$ however, distinguishes between the driven (IDLG, RDLG) and the equilibrium (LG) lattice gases. In order to understand this, we first write the phenomenological scaling form for $O$, analogous to Eq. \eqref{eq:scalingass} in the Letter, \begin{eqnarray} O (t, L_{\parallel} ; S_\Delta) = L_{\parallel}^{-\beta/[\nu(1+\Delta)]} \tilde f_O (t/L_{\parallel...
```

Outputs:

- full_prefill (0.23s)
```text
$m \sim t^{1/2}$ and $O \sim t^{\alpha} L_\parallel^{-1/2}$ with $\alpha = 1/10$
```
- full_kv_reuse (0.02s)
```text
(a $n LG and $n LG.
```
- droidspeak (0.11s)
```text
The dynamical behaviour of $O$ in Lattice Gases with either constant (IDLG) or random (RDLG) and equilibrium (LG) and equilibrium (LG) and equilibrium (LG) and equilibrium (LG) and equilibrium (LG) and equilibrium (LG)
```
- cacheblend_reference (0.22s)
```text
\footnote: $n the lowest mode, $nays, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n the lowest mode, $n
```
