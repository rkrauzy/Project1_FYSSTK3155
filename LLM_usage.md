# LLM Usage Declaration

This file documents the use of large language models in the development of the code for Project 1 in FYS-STK3155/4155.

The project authors reviewed, tested and interpreted all submitted code and results. LLM assistance is classified according to the course guidelines using Levels 0--4.

## Shared source files

### `src/data.py`

**LLM level:** 0 - None

The implementation of the Runge function, data generation and polynomial design matrix was written independently without LLM assistance.

---

### `src/models.py`

**LLM level:** 0 - None

The implementations of the closed-form OLS and Ridge regression estimators were written independently without LLM assistance.

---

### `src/metrics.py`

**LLM level:** 0 - None

The implementations of mean squared error and \(R^2\) were written independently without LLM assistance.

---

### `src/gradient_descent.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT substantially assisted with the formulation and implementation of the OLS and Ridge cost functions, analytical gradients and the reusable fixed-learning-rate gradient descent routine. Assistance also included treatment of the Ridge intercept and storage of the parameter history.

**Verification:** The implementation was reviewed and executed by the project authors. The analytical gradients were compared with JAX automatic differentiation to machine precision, and the resulting gradient descent solutions were compared with the closed-form OLS and Ridge solutions.

---

## Part A

### `parts/part_a/baseline.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** ChatGPT assisted with the overall structure of the baseline OLS experiment, including the train/test workflow, evaluation across polynomial degrees and presentation of MSE and \(R^2\).

**Verification:** Reviewed, executed and adapted by the project authors.

---

### `parts/part_a/sample_size.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** ChatGPT assisted with smaller code sections for comparing OLS performance across different sample sizes.

**Verification:** Reviewed and executed by the project authors.

---

### `parts/part_a/noise.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** ChatGPT assisted with smaller code sections for investigating how different noise levels affect OLS regression performance.

**Verification:** Reviewed and executed by the project authors.

---

### `parts/part_a/parameters.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** ChatGPT assisted with the overall structure for examining and visualising the fitted OLS coefficients as polynomial degree increases.

**Verification:** Reviewed, executed and interpreted by the project authors.

---

### `parts/part_a/fitted_curves.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT suggested and substantially assisted with an additional visualisation comparing OLS polynomial fits with the true Runge function. The suggestion was reviewed and retained by the project authors to provide additional visual evidence for the behaviour and instability of higher-degree polynomial models.

**Verification:** Reviewed, executed and interpreted by the project authors.

---

### `parts/part_a/conditioning.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT suggested and substantially assisted with an additional condition-number analysis comparing centered and uncentered polynomial design matrices. The suggestion was reviewed and retained by the project authors to provide quantitative support for the discussion of centering and numerical stability.

**Verification:** Reviewed, executed and interpreted by the project authors.

---

## Part B

### `parts/part_b/ridge_baseline.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** ChatGPT assisted with smaller code sections used to compare Ridge regression with the OLS baseline for different regularisation strengths.

**Verification:** Reviewed and executed by the project authors.

---

### `parts/part_b/ridge_parameters.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** ChatGPT assisted with smaller code sections for examining how Ridge regularisation changes the fitted polynomial coefficients.

**Verification:** Reviewed and executed by the project authors.

---

### `parts/part_b/ridge_singular_values.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** ChatGPT assisted with smaller code sections for analysing the singular values associated with the polynomial regression problem.

**Verification:** Reviewed and executed by the project authors.

---

### `parts/part_b/ridge_robustness.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT suggested and substantially assisted with an additional robustness analysis of Ridge regression across different sample sizes and noise levels. The suggestion was reviewed and retained by the project authors to strengthen the empirical analysis and compare the results with Part A.

**Verification:** Reviewed, executed and interpreted by the project authors.

---

## Part C

### notebooks/partc_theory.ipynb
#### Derivation of bias variance

**Tool:** Claude Sonnet 5.5 and Opus 5 (Anthropic), September 2026

**LLM level:** 2 - Editorial and 3 — Generative

**Contribution:** 

Level 2: The derivation itself (the decomposition into A, B, C, the
treatment of each expectation value, and the vanishing of the cross terms) was
dictated step by step by the author; Claude converted it to LaTeX and corrected
two sign errors in the process. Claude also proposed the overall structure of the section
(assumptions first, interpretation last).

Level 3: For the final step (going from a single test point to the average over all $n$ points, and matching the notation of the exercise), Claude first checked the author's single-point derivation (no errors found). The author noticed that his left-hand side, $\frac1n\sum_i\mathbb E[(y_i-\tilde y_i)^2]$, did not match the exercise's, $\frac1n\sum_i(y_i-\tilde y_i)^2=\mathbb E[(\boldsymbol y-\tilde{\boldsymbol y})^2]$. Claude explained that the exercise uses $\mathbb E$ both as a sample average and as an expectation over the training set $\mathcal L$ and the noise $\boldsymbol\varepsilon$, and drafted the paragraph that resolves this: the cost $C$ is a random variable, the decomposition concerns $\mathbb E_{\mathcal L,\varepsilon}[C]$ (by linearity), and $\mathbb E[\tilde{\boldsymbol y}]$ is the vector of per-point means over training sets. Claude also wrote the LaTeX of the closing equations (rewriting the result as $\mathrm{Bias}^2[\tilde{\boldsymbol y}]+\mathrm{var}[\tilde{\boldsymbol y}]+\sigma^2$) and the remark on how the expectations are estimated in practice (bootstrap). The author compared this with his own derivation and the exercise text before including it.

**Verification:** The author checked every step against Eqs. (2.49)–(2.52) of
the lecture notes, including the two corrected sign errors, and confirmed that
each assumption used to eliminate the cross terms is stated explicitly in the
opening paragraph.

#### Absorbtion of the noise variance

**Tool:** Claude Sonnet 5.5 (Anthropic), September 2026

**LLM level:** 2 - Editorial

**Contribution:** The approach was proposed by the author: rather than proving
the result separately, the noise term $\varepsilon_0$ is regrouped with $f_0$ in
the decomposition already derived above, so that $(f_0 + \varepsilon_0) = y_0$
and the same expansion can be reused with two terms $a$ and $b$ instead of
three. The author dictated the accompanying text and every step of the
derivation, including the evaluation of $\mathbb{E}[a^2]$, $\mathbb{E}[b^2]$ and
the vanishing of $\mathbb{E}[2ab]$. Claude converted the dictation to LaTeX and
corrected three errors in it: two sign errors in the expansion of
$\mathbb{E}[a^2]$ and a missing expectation operator on the right-hand side of the
first line.

**Verification:** Since this derivation is a regrouping of the one in the
preceding subsection, the author checked that the two agree term by term. The
result $\mathbb{E}[a^2] = (f_0-\mu_0)^2 + \sigma^2$ matches the statement in
the project description that the measured bias absorbs the noise variance.

#### Interpretation of the tree terms

**Tool:** Claude Sonnet5.5 (Anthropic), September 2026

**LLM level:** 3 - Generative

**Contribution:** The substance of this subsection was settled through extended
discussion between the author and Claude before any text was written. In the
course of that discussion the author worked through the interpretation of the
three terms and the distinction between the randomness appearing in the
derivation and the randomness the bootstrap actually realises. The author also
identified that the lecture notes use $\mathbb{E}$ for two different operations
— the average over test points in Eq. (2.50) and the expectation over training
sets in Eq. (2.52) — and revised his own earlier week 36 solution, in which the
bootstrap identity error $=$ bias $+$ variance had been described as an
inequality rather than as an exact algebraic identity. Claude then drafted both
paragraphs from the author's instructions regarding content and length. The
formulations are Claude's; the content had been established beforehand, and the
author read, checked and endorsed the result.

**Verification:**

---

## Part E

### `parts/part_e/gradient_check.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT substantially assisted with the implementation of gradient verification using JAX automatic differentiation for OLS and Ridge.

**Verification:** Reviewed and executed by the project authors, with analytical and automatic gradients compared numerically to machine precision.

---

### `parts/part_e/convergence.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** ChatGPT assisted with the overall structure of the convergence analysis, including comparison of gradient descent with closed-form OLS and Ridge solutions, error tracking and plotting.

**Verification:** Reviewed, executed and adapted by the project authors.

---

### `parts/part_e/learning_rate.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** ChatGPT assisted with the overall structure of the learning-rate analysis, including Hessian-based stability limits, comparison of several learning rates, error tracking and plotting for OLS and Ridge.

**Verification:** Reviewed, executed and adapted by the project authors.

---

### `parts/part_e/autodiff_gd.py`

**Tool:** ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** ChatGPT substantially assisted with the implementation of gradient descent using both analytical gradients and JAX automatic differentiation for OLS and Ridge, including convergence comparisons with closed-form solutions.

**Verification:** Reviewed and executed by the project authors, with analytical and JAX-based results compared numerically and visually.

