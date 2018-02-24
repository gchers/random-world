# random-world [![Build Status](https://travis-ci.org/gchers/random-world.svg?branch=master)](https://travis-ci.org/gchers/random-world) ![Version](https://img.shields.io/crates/v/random-world.svg)

This is a `rust` implementation of Machine Learning (ML) methods for confident
prediction (e.g., Conformal Predictors) and related ones introduced in the book
Algorithmic Learning in a Random World ([ALRW](http://alrw.net/)).

## Goals

These are the main goals of this library.
The fact that something appears here does not imply that it has already been
fulfilled.

- Fast implementation of methods described in ALRW book
- Be well documented
- CP method should be able to use scoring classifiers from existing
  ML libraries (e.g., [rusty-machine](https://athemathmo.github.io/rusty-machine/doc/rusty_machine/), [rustlearn](https://maciejkula.github.io/rustlearn/doc/rustlearn/))
- Easily interface with other languages (e.g., Python)
- Provide standalone programs for some functions (e.g., in the way
  libsvm does for SVM methods)


## Install

Include the following in _Cargo.toml_:

```
[dependencies]
random-world = "0.2.1"
```

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

// Create a k-NN nonconformity measure (k=2)
let ncm = KNN::new(2);
// Create a Conformal Predictor with the chosen nonconformity
// measure and significance level 0.3.
let mut cp = CP::new(ncm, Some(0.3));

// Create a dataset
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

Please, read the [docs](https://docs.rs/random-world/0.1.0/random_world/) for
more examples.

## Standalone Binaries

*random-world* provides standalone binaries for the main functionalities.

To install them, first install Rust's package manager
[cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html), then run:

```
cargo install random-world
```

### cp-predict

`cp-predict` runs CP on a training set, uses it to predict a test set;
each dataset should be contained in a CSV file with rows:

    label, x1, x2, ...

where `label` is a label id, and x1, x2, ...
are the values forming a feature vector.
It is important that label ids are `0, 1, ..., n_labels-1` (i.e., with no
missing values); one can specify  `--n-labels` if not all labels
are available in the initial training data (e.g., in on-line mode).

Results are returned in a CSV file with rows:

    p1, p2, ...

where each value is either a prediction (true/false) or
a p-value (float in [0,1]), depending on the chosen output;
each row contains $L$ values, each one corresponding to
one label.

Example:
```
$ cp-predict knn -k 1 predictions.csv train_data.csv test_data.csv
```
Runs CP with nonconformity measure k-NN (k=1) on `train_data.csv`,
predicts `test_data.csv`, and stores the output into
`predictions.csv`.
The default output are p-values; to output actual predictions, specify
a significance level with `--epsilon`.

To run CP in on-line mode on a dataset (i.e., predict one object
per time and then append it to the training examples), only specify
the training file:
```
$ cp-predict knn -k 1 predictions.csv train_data.csv
```

More options are documented in the help:
```
$ cp-predict -h
```

## Features

Methods:
- [x] Deterministic and smoothed Conformal Predictors (aka, transductive CP)
- [ ] Deterministic and smoothed Inductive Conformal Predictors (ICP)
- [x] Plug-in and Power martingales for exchangeability testing
- [ ] Venn Predictors

Nonconformity measures:
- [x] k-NN
- [ ] KDE
- [ ] Generic wrapper around ML scorers from existing libraries (e.g., rusty-machine)

Bindings:
- [ ] Python bindings

Binaries:
- [x] CP (both batch prediction and on-line)
- [x] Martingales

## Authors

* Giovanni Cherubin ([giocher.com](https://giocher.com))

## Similar Projects

- [nonconformist](https://github.com/donlnz/nonconformist/) is a well
  maintained Python implementation of CP and ICP.
