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
