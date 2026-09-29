# LLM Usage Declaration

This file documents the use of large language models in the development of the code for Project 1 in FYS-STK3155/4155.

The project authors reviewed, tested and interpreted all submitted code and results. LLM assistance is classified according to the course guidelines using Levels 0--4.


### Repository restructuring and fixes

**Tool:** Claude (Claude Code, Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** Parts a), b) and e) were first written as standalone scripts with an intercept column in the design matrix. When the group agreed on a shared repository structure, Claude Code was used to adapt the code to it. In `src/data.py`, the intercept column was removed from `design_matrix`, `scale_matrix` was updated to standardise all columns, `center_y` was added and a global `SEED` constant was defined. In `src/models.py`, `ridge` was updated to the closed-form solution (XᵀX + nλI)⁻¹Xᵀy. In `src/gradient_descent.py`, `ridge_cost` and `ridge_gradient` were updated to regularise all parameters uniformly. In `src/plotting.py`, `dpi=150` and `plt.close()` were added to `save_fig`. The scripts in `parts/part_a/` and `parts/part_b/` were rewritten to import from the shared `src/` modules.

Claude Code was later used to fix the part e) and f) scripts so they match the same structure. In the part e) scripts, the manual centering of `X_train[:, 1:]` was replaced with `scale_matrix`, and `y_train` is centred with `center_y`, as in `resampling.py`. `ridge_cost_jax` now matches `models.ridge`, with the penalty `lmbda * sum(theta**2)` on all coefficients, and `H_ridge` is now (2/n)XᵀX + 2λI, the Hessian of that cost. The part f) Optax scripts now use the same seed, `split_data`, `scale_matrix` and `center_y` as part e), so the results are comparable. Figures are saved with `save_fig` instead of `plt.show()`, with separate OLS and Ridge Optax convergence plots. Inaccurate claims about intercept exclusion and a machine-precision check were removed from earlier LLM declarations.

**Verification:** All changes were reviewed and executed by the project authors. The restructured scripts were checked against the original standalone results, the analytical and JAX Ridge gradients were compared to machine precision with `gradient_check.py`, and all part e) and f) figures were regenerated.


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

### `src/resampling.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** The bootstrap function is based on the bootstrap code in the week 36 Tuesday notebook (`week36tuesday.ipynb`). Claude assisted with adapting it to the repository structure: building the design matrix with `design_matrix`, scaling and centering each bootstrap sample with `scale_matrix` and `center_y` since the design matrix has no intercept column, and reshaping `y_test` to a column vector so that the error, bias and variance expressions broadcast correctly. The project authors replaced the least squares solver with their own `ols` function and corrected the indexing of the degree arrays.

**Verification:** Reviewed and executed by the project authors. The bootstrap estimates were checked to satisfy error = bias + variance to machine precision, and the results were compared with the original notebook version.

---

### `src/optimiser.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 2 - Snippet

**Contribution:** The functions `optimiser_step`, `make_batches`, `step_length` and `sgd` are our own code from the week 38 exercises (`week38.ipynb`, Exercises 2 and 4), which follow the section Implementations in Chapter 4 of the lecture notes. Claude assisted with moving them to `src/optimiser.py` so they can be reused in parts h) and i), and with adapting `sgd` to the repository structure: the notebook's gradient function was replaced by `ridge_gradient` or `lasso_gradient` from `src/gradient_descent.py`, chosen by the arguments `lmbda` and `lasso`, and the default seed was set to `SEED` from `src/data.py`. The function `optimise_optax` is not covered by this entry.

**Verification:** Reviewed and executed by the project authors. With `method="plain"` and a batch size equal to the number of training points, `sgd` reproduces the closed-form OLS and Ridge solutions to $10^{-14}$.

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

### `parts/part_c/train_test_MSE.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 4 - Substantial

**Contribution:** Claude substantially assisted with the implementation of the training and test MSE analysis reproducing Fig. 2.11 of Hastie et al. After a single train/test split gave a noisy test MSE, Claude suggested averaging the training and test MSE over 500 data sets with new noise and new splits, in line with Fig. 7.1 of Hastie et al., and presenting the results for $n = 40$, $100$ and $400$ in one figure. The subplot structure follows Step 2 of the week 36 Tuesday notebook. The suggestion was reviewed and retained by the project authors.

**Verification:** Reviewed, executed and interpreted by the project authors. The averaged results were compared with single split results to confirm that the difference is due to the variance of the test MSE for small test sets.

### `parts/part_c/bias_variance.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** Claude assisted with the overall structure of the bias-variance analysis using the shared `bootstrap` function, including the comparison of $n = 40$, $100$ and $400$ in one figure and the reference line for the noise variance $\sigma^2$. The structure follows Steps 2 and 3 of the week 36 Tuesday notebook (week36tuesday.ipynb)

**Verification:** Reviewed, executed and interpreted by the project authors.

---
## Part D

### `parts/part_d/cv_ols.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** The cross-validation code follows Step 4 of Case 2 in the week 36 Tuesday notebook (`week36tuesday.ipynb`), using `KFold` and `cross_val_score`. Claude assisted with adapting it to the repository structure: using `design_matrix` from `src/data.py`, placing `StandardScaler` inside the pipeline so that scaling is fitted on the training folds only, and using `LinearRegression` with an intercept since the design matrix has no intercept column. Claude also assisted with running the analysis for both $k = 5$ and $k = 10$ and with plotting the cross-validated MSE together with the bootstrap error from Part C in one figure for comparison.

**Verification:** Reviewed, executed and interpreted by the project authors.

---

### `parts/part_d/cv_ridge.py`

**Tool:** Claude, Claude Opus 5.5 (Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** The cross-validation code follows Steps 2 and 3 of Case 2 in the week 36 Tuesday notebook (`week36tuesday.ipynb`), where the cross-validated MSE for Ridge is computed with `KFold` and `cross_val_score` for a fixed polynomial degree. Claude assisted with extending this to a grid over both polynomial degree and $\lambda$ by adding an outer loop over the degree, with adapting it to the repository structure (`design_matrix` from `src/data.py`, `StandardScaler` inside the pipeline so that scaling is fitted on the training folds only), and with choosing a range of $\lambda$ suited to the scale of the Runge data. Claude also suggested presenting the resulting grid as heatmaps for $k = 5$ and $k = 10$, and assisted with the interpretation of the results.
    
Also Claude spotted minor inconsistency and fixed it: Since in cv_own_vc_sklearn.py
we use ridge from src.models/ which uses alpha = n * lmbdas we got different scales for lambda. Adjusted for that.   

**Verification:** Reviewed, executed and interpreted by the project authors.

---

### `parts/part_d/cv_own_vs_sklearn.py`

**Tool:** Claude, Claude Opus 5.5 (Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** The script follows Steps 1 and 2 of Case 2 in the week 36 Tuesday notebook (`week36tuesday.ipynb`), where an own $k$-fold loop for Ridge regression is compared with `cross_val_score` using the same `KFold` object. Claude assisted with adapting the own loop to the repository structure, replacing `PolynomialFeatures`, `StandardScaler` and `Ridge` with `design_matrix`, `scale_matrix`, `center_y` and the project's own `ridge` function, and with matching the regularisation convention by using `alpha = n_train * lambda` in `Scikit-Learn`, since the project's `ridge` includes the factor $1/n$ in the cost function. Claude also assisted with the choice of polynomial degree and range of $\lambda$, and with the interpretation of the results.

**Verification:** Reviewed, executed and interpreted by the project authors. The own loop and `cross_val_score` agree to a relative difference of order $10^{-10}$ for all  $\lambdas$.

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

---

## Part G

### `parts/part_g/lasso_gd.py`

**Tool:** Claude, Opus 5.5 (Anthropic, September 2026)

**LLM level:** 3 - Skeleton

**Contribution:** The script solves the Lasso problem, Eq. (3.57) in the lecture notes, with the gradient descent methods from parts e) and f), using `lasso_gradient` from `src/gradient_descent.py`. Claude assisted with assembling it from existing code: the gradient check from `parts/part_e/gradient_check.py`, extended to θ = 0 where |θ| is not differentiable; plain gradient descent with η = 1/λ_max of the OLS Hessian from `parts/part_e/convergence.py`; the momentum, AdaGrad, RMSprop and Adam runs with `optimise_optax` from `parts/part_f/convergence_optax.py`; the reference solution from `lasso_fit` in Chapter 3 of the lecture notes, with `alpha = λ/2` since `Scikit-Learn`'s Lasso divides the squared error by $2n$; and the test predictions from `parts/part_d/cv_own_vs_sklearn.py`. Claude also checked that `jax.grad` returns 1 for the derivative of |θ| at zero, while `np.sign` returns 0.

**Verification:** Reviewed and executed by the project authors. The analytical and JAX gradients agree to $10^{-16}$ away from θ = 0 and differ by exactly λ at θ = 0, and all five methods reach a Lasso cost within $2 \cdot 10^{-4}$ of the `Scikit-Learn` solution.
