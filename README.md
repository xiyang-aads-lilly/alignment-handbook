# The Alignment Handbook (Lilly Version)

Check out the `scripts` and `recipes` directories for instructions on how to train some models!

## Project structure
```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── chapters                    <- Educational content to render on hf.co/learn
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── experiment                  <- Some demo experiment scripts
├── scripts                     <- Scripts to train and evaluate chat models
├── setup.cfg                   <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

## How to use with Magtrain
- check `experiment` directory to see examples on how to run single / multi-node training on Magtrain
