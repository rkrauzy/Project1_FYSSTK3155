# Project1_FYSSTK3155


## midlertidig kommentar til gutta som forklarer

### Repo struktur: 

Vi forsøker å ha all gjenbrukbar kode i src (source) så når f.eks albert skal gjøre oppgave c har jeg ferdig data som kan hentes ut og kan bare kalle på modellene jeg trenger, f.eks OLS eller Lasso Hvis jeg skulle trenge å resample i oppgaven min ligger også koden i resampling.py og jeg kan bare kalle på det. 

Når man løser en oppgave kan det gjøre inni  for eksempel part_c/ mappa og 


```
Project1_FYSSTK3155/
├── README.md                # hva prosjektet er, hvordan kjøre koden
├── LLM_usage.md             # løpende logg over LLM-bruk
├── requirements.txt         # pakker som må installeres
├── .gitignore
│
├── src/                     # gjenbrukbar kode som importeres
│   ├── __init__.py
│   ├── data.py              # Runge-funksjon, støy, skalering, train/test split
│   ├── metrics.py           # mse, r2
│   ├── models.py            # OLS, Ridge, Lasso
│   ├── gradient_descent.py  # GD, momentum, AdaGrad, RMSprop, Adam, SGD
│   ├── resampling.py        # bootstrap, k-fold CV
│   └── plotting.py          # lagrer alle plott i figures/
│
├── parts/                   # én mappe per deloppgave
│   ├── __init__.py
│   ├── part_a/
│   │   └── fil.py
│   ├── part_b/
│   │   └── fil.py
│   └── ...
│
├── notebooks/               # utforsking og eksperimenter, valgfritt
└── figures/                 # alle ferdige plott, lastes opp til Overleaf
```

## Forklaring av ulike mapper: 
### (1) LLM_usage.md
Hver gang man genererer med claude kode skal man manuelt fylle inn eksempelvis: 

## Code (levels 0-4)
| File | Level | Who | Description |
|---|---|---|---|
| src/data.py | 0 | Albert | Written independently |
| src/gradient_descent.py | 3 | ... | Claude gave the class skeleton; update rules written and tested by us |

## Text (levels 0-3)
| Section | Level | Notes |
|---|---|---|
| Abstract | 1 | Grammar check |

Her er link til hva som er ulike levels:
https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/LLM_Usage_Declaration_Guidelines.md

BTW.. i funksjoner dokumenterer man AI bruk slik: 
def heipådeg():
    """

    LLM-assisted
    ------------
    Tool: GitHub Copilot (March 2026)
    Role: Generated the Jacobian accumulation loop using torch.autograd.functional.jacobian.
    Modifications: Added batching over inputs to avoid OOM on GPU; verified output against
    finite-difference approximation on a two-layer network.
    """
    pass

### (2) requirements.txt
Her legger man inn pakker man må installere for å kjøre programmet, eksempelvis under:

numpy
scipy
matplotlib
scikit-learn
jax
jupyter

I terminal skriver man bare "pip install -r requirements.txt"

### (3) data.py
Ralph skal fikse denne, men generelt sett er denne:

SEED = 4155

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def make_data(n, sigma=, rng=None):
    ..
def poly_design(x, degree):
    ..

### (4) metrics.py
Ulike funksjoner man kan kalle på hvis det trengs. veldig enkel fil som definerer og anvender MSE eller MAE osv. 

### (5) models.py
Også veldig enkel fil som anvender OLS, RIdge og Lasso. 

### (6) data VIKTIG
I overleaf skal man laste opp en mappe med alle bilder/plots man vil bruke. Den skal inneholde bare .png filer som man enkelt kan legge inn i rapporten. 
Eksempelvis når jeg har skrevet kode og ønsker å lage et bilde skriver man kode slik: 

from src.plotting import save_fig

plt.plot(degrees, mse_test)  
plt.xlabel("Polynomial degree")  
plt.ylabel("MSE")  
save_fig("partA_mse_vs_degree")   
plt.close()          # frigjør minne når et script lager mange figurer. 

Her har man laget et plot og kaller det for "partA_mse_vs_degree". Det vil lagres i figures/ mappen i repoet sammen med resten av bildene. Plottet blir lagret i mappen som en .png. 


