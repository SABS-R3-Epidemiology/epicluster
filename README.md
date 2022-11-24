# epicluster

This repository contains a Python package that can be used to estimate changes in the time-varying reproduction number from time series of cases. The statistical model is from the field of Bayesian nonparametrics, and results from using this package are given in [Creswell et al., 2022, "A Bayesian nonparametric method for detecting rapid changes in disease transmission", Journal of Theoretical Biology](https://www.sciencedirect.com/science/article/pii/S0022519322003423).

## Installation

Local copies of the package files are installable via `pip`:

```bash
pip install -e .
```

## Usage

See the examples directory for a simple notebook performing inference. 

The results repository (https://github.com/SABS-R3-Epidemiology/epicluster-results) contains multiple examples illustrating the full functionality of the package.


## References
The model of change points is based on:

[1]
Martínez, A. F., & Mena, R. H. (2014). On a nonparametric change point detection model in Markovian regimes. Bayesian Analysis, 9(4), 823-858.

The epidemiological model is based on:

[2]
Thompson RN, Stockwin JE, van Gaalen RD, Polonsky JA, Kamvar ZN, Demarsh PA,
Dahlqwist E, Li S, Miguel E, Jombart T, Lessler J. (2019). Improved inference of
time-varying reproduction numbers during infectious disease outbreaks.
Epidemics 29: 100356.

[3]
Creswell R, Augustin D, Bouros I, Farm HJ, Miao S, Ahern A, Robinson M, Lemenuel-Diot A,
Gavaghan DJ, Lambert B, Thompson RN (2022). Heterogeneity in the onwards tranmission risk between local and imported cases affects practical estimates of the time-dependent reproduction number. Philosophical Transactions of the Royal Society A (forthcoming).
