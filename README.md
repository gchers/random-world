# random-world

This is a `rust` implementation of Machine Learning (ML) methods for confident
prediction (e.g., Conformal Predictors) and related ones introduced in the book
Algorithmic Learning in a Random World ([ALRW](http://alrw.net/)).

## Goals

These are the main goals of this library.
The fact that something appears here does not imply that it has already been
fulfilled.

* Fast implementation of methods described in ALRW book
* Be well documented
* CP method should be able to use scoring classifiers from existing
  ML libraries (e.g., [rusty-machine](https://athemathmo.github.io/rusty-machine/doc/rusty_machine/), [rustlearn](https://maciejkula.github.io/rustlearn/doc/rustlearn/))
* Easily interface with other languages (e.g., Python)
- Provide standalone programs for some functions (e.g., in the way
  libsvm does for SVM methods)


## Install

Using cargo, as soon as this code is published as a crate on
[crates.io](crates.io).

## Quick Intro

Using a deterministic (i.e., non smooth) Conformal Predictor with k-NN
nonconformity measure (`k=2`) and significance level `epsilon=0.3`.
The prediction region will contain the correct label with probability
`1-epsilon`.

```rust
#[macro_use(array)]
extern crate ndarray;
extern crate random_world;

use random_world::cp::*;
use random_world::ncm::*;

let ncm = KNN::new(2);
let mut cp = CP::new(ncm, Some(0.3));
let train_inputs = array![[0., 0.],
                          [1., 0.],
                          [0., 1.],
                          [1., 1.],
                          [2., 2.],
                          [1., 2.]];
let train_targets = array![0, 0, 0, 1, 1, 1];
let test_inputs = array![[2., 1.],
                         [2., 2.]];

// Train and predict
cp.train(&train_inputs.view(), &train_targets.view())
  .expect("Failed prediction");
let preds = cp.predict(&test_inputs.view())
              .expect("Failed to predict");
assert!(preds == array![[false, true],
                        [false, true]]);
```
More examples on deterministic/smooth Conformal Predictors at
[CP](/cp/cp/struct.CP.html).


## Features

Methods:
- [x] Deterministic and smoothed Conformal Predictors (aka, transductive CP)
- [ ] Deterministic and smoothed Inductive Conformal Predictors
- [ ] Martingales
- [ ] Venn Predictors

Nonconformity measures:
- [x] k-NN
- [ ] KDE
- [ ] Generic wrapper around existing ML scorers (e.g., rusty-machine)

Bindings:
- [ ] Python bindings

Binaries:
- [ ] CP
- [ ] Martingales

## Authors

* Giovanni Cherubin [giocher.com](https://giocher.com)

## License
