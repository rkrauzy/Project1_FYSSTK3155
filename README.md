# FYS-STK3155 Project 1: Regression and gradient methods for the Runge function

### Authors: 
- Albert Sjåvåg
- Teodor Aursnes
- Raphael Katsumi Soma Rauzy
- Jørgen Sannerhaugen Florholmen

University of Oslo, autumn 2026

We fit the Runge function $f(x) = 1/(1+25x^2)$ on $[-1, 1]$ with polynomials, using ordinary least squares (OLS), Ridge and Lasso regression. Model complexity and the penalty are chosen with the bootstrap and $k$-fold cross-validation, and the closed-form solutions are compared with our own gradient descent, momentum, AdaGrad, RMSprop, Adam and stochastic gradient descent. The report is submitted separately as a PDF.

## Installation

Tested with Python 3.13.

```
pip install -r requirements.txt
```

## Running the code

All scripts are run as modules from the repository root:

```
python -m parts.part_c.bias_variance
```

Figures are saved to `figures/`. Most scripts finish in a few seconds. `parts.part_i.cv_lasso` takes about 3 minutes and `parts.part_i.bias_variance_lambda` about 1 minute. All random numbers use the seed 2026 (`SEED` in `src/data.py`), so every run reproduces the numbers in the report.

The printed output of every script is stored in `outputs/`, so the numbers in the report and in `tables/` can be checked without running the code. To regenerate one file, for example:

```
python -m parts.part_c.bias_variance > outputs/parts.part_c.bias_variance.txt 2>/dev/null
```

## Structure

```
src/                     reusable code, imported by all scripts
  data.py                Runge function, data generation, design matrix, scaling, train/test split
  models.py              closed-form OLS and Ridge; Lasso via Scikit-Learn
  gradient_descent.py    cost functions, analytical gradients (OLS, Ridge, Lasso), plain gradient descent
  optimiser.py           momentum, AdaGrad, RMSprop, Adam and minibatch SGD (own code); Optax wrapper
  resampling.py          bootstrap bias-variance decomposition, over degree and over lambda
  plotting.py            save_fig, saves every figure to figures/
parts/part_a … part_i/   one folder per part of the project, one script per analysis
figures/                 all figures used in the report
tables/                  LaTeX tables with the numbers printed by the scripts
outputs/                 printed output of every script, e.g. parts.part_c.bias_variance.txt
LLM_usage.md             declaration of the use of large language models
```

## Scripts, figures and tables

The output of each script is in `outputs/parts.<part>.<script>.txt`.

| Part | Script | Figure(s) in `figures/` | Table in `tables/` |
|---|---|---|---|
| a) OLS | `part_a/baseline.py` | `ols_mse` | |
| | `part_a/parameters.py` | `ols_parameters` | |
| | `part_a/fitted_curves.py` | `ols_fitted_curves` | |
| | `part_a/conditioning.py` | `ols_conditioning` | |
| | `part_a/sample_size.py` | `ols_sample_size` | |
| | `part_a/noise.py` | `ols_noise` | |
| b) Ridge | `part_b/ridge_baseline.py` | `ridge_mse` | |
| | `part_b/ridge_parameters.py` | `ridge_parameters` | |
| | `part_b/ridge_singular_values.py` | `ridge_singular_values` | |
| | `part_b/ridge_robustness.py` | `ridge_robustness_n`, `ridge_robustness_sigma` | |
| c) Bias-variance | `part_c/train_test_MSE.py` | `prediction_error_vs_complexity` | `train_test_MSE` |
| | `part_c/bias_variance.py` | `bias_variance_tradeoff` | `bias_variance` |
| d) Cross-validation | `part_d/cv_ols.py` | `cv_vs_bootstrap_ols` | `cv_ols` |
| | `part_d/cv_ridge.py` | `cv_ridge_heatmap` | `cv_ridge` |
| | `part_d/cv_own_vs_sklearn.py` | `cv_own_vs_sklearn` | `cv_own_vs_sklearn` |
| e) Gradient descent | `part_e/gradient_check.py` | | `gradient_check` |
| | `part_e/autodiff_gd.py` | `autodiff_gd_ols`, `autodiff_gd_ridge` | `autodiff_gd` |
| | `part_e/convergence.py` | `gd_convergence` | `convergence` |
| | `part_e/learning_rate.py` | `gd_learning_rate_ols`, `gd_learning_rate_ridge` | `learning_rate` |
| f) Adaptive methods | `part_f/convergence_optax.py` | `convergence_optax_ols`, `convergence_optax_ridge` | `convergence_optax` |
| | `part_f/learning_rate_optax.py` | `Learning_rate_optax_methods` | `learning_rate_optax` |
| g) Lasso | `part_g/lasso_gd.py` | | `lasso_gd` |
| h) SGD | `part_h/sgd_methods.py` | | `sgd_methods` |
| | `part_h/sgd_study.py` | `sgd_study` | `sgd_study` |
| i) Model selection | `part_i/cv_lasso.py` | `cv_lasso_heatmap` | `cv_lasso` |
| | `part_i/model_selection.py` | `model_selection` | `model_selection` |
| | `part_i/bias_variance_lambda.py` | `bias_variance_lambda` | `bias_variance_lambda` |

## Conventions

- The design matrix has no intercept column. The columns of $X$ are standardised and $y$ is centred with the training statistics only, also inside every cross-validation fold and bootstrap sample.
- Ridge minimises $(1/n)\|X\theta - y\|^2 + \lambda\|\theta\|_2^2$, with closed form $(X^TX + n\lambda I)^{-1}X^Ty$. Scikit-Learn's `Ridge` therefore uses `alpha = n * lambda`.
- Lasso minimises $(1/n)\|X\theta - y\|^2 + \lambda\|\theta\|_1$. Scikit-Learn's `Lasso` therefore uses `alpha = lambda / 2`.
- 100 data points, noise $\sigma = 0.1$, 80/20 train/test split.

## Verification

- Analytical gradients agree with JAX automatic differentiation to $10^{-16}$ (OLS, Ridge, and Lasso away from $\theta_j = 0$).
- Gradient descent and full-batch SGD reproduce the closed-form OLS and Ridge solutions to $10^{-14}$.
- Our own $k$-fold loop reproduces `cross_val_score` to a relative difference of $3 \times 10^{-10}$.
- The bootstrap estimates satisfy error = bias$^2$ + variance to machine precision.
- Gradient descent on the Lasso cost reaches the cost of Scikit-Learn's solution to within $2 \times 10^{-4}$.

## Use of large language models

See [LLM_usage.md](LLM_usage.md) for a file-by-file declaration.
