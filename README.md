# random-world [![Build Status](https://travis-ci.org/gchers/random-world.svg?branch=master)](https://travis-ci.org/gchers/random-world) ![Version](https://img.shields.io/crates/v/random-world.svg)

This is a `rust` implementation of Machine Learning (ML) methods for confident
prediction (e.g., Conformal Predictors) and related ones introduced in the book
Algorithmic Learning in a Random World ([ALRW](http://alrw.net/)),
which also provides standalone binaries.

## Goals

These are the main goals of this library.

- Fast implementation of methods described in ALRW book
- Be well documented
- Provide standalone binaries
- CP method should be able to use scoring classifiers from existing
  ML libraries (e.g., [rusty-machine](https://athemathmo.github.io/rusty-machine/doc/rusty_machine/), [rustlearn](https://maciejkula.github.io/rustlearn/doc/rustlearn/))
- Easily interface with other languages (e.g., Python)

# Binaries

Standalone binaries are meant to cover most functionalities of the library.
They operate on `.csv` files, and allow to make CP predictions, test exchangeability,
and much more.

## Installation

To install the binaries, install Rust's package manager
[cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).
Then run:

```
cargo install random-world
```

This will install on your system the binaries: `cp-predict`, `icp-predict`,
and `martingales`.

## cp-predict

`cp-predict` allows making *batch* and *on-line* CP predictions.
It runs CP on a training set, and uses it to predict a test set;
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
a p-value (float in [0,1]), depending on the flags passed to `cp-predict`;
each row contains a value for each label.

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

Predict data using Conformal Prediction.

If no <testing-file> is specified, on-line mode is assumed.

Usage: cp knn [--knn=<k>] [options] [--] <output-file> <training-file> [<testing-file>]
       cp kde [--kernel<kernel>] [--bandwidth=<bw>] [options] [--] <output-file> <training-file> [<testing-file>]
       cp (--help | --version)

Options:
    -e, --epsilon=<epsilon>     Significance level. If specified, the output are
                                label predictions rather than p-values.
    -s, --smooth                Smooth CP.
    --seed=<s>                   PRNG seed. Only used if --smooth set.
    -k, --knn=<kn>              Number of neighbors for k-NN [default: 5].
    --n-labels=<n>              Number of labels. If specified in advance it
                                slightly improves performances.
    -h, --help                  Show help.
    --version                   Show the version.
```

## icp-predict

The syntax for `icp-predict` is currently identical to that of `cp-predict`:
the size of the calibration set is chosen to be half the size of the training
set.
This will change (hopefully soon).

## martingales

Computes exchangeability martingales from a file of p-values.
P-values should be computed using `cp-predict` in an on-line setting
for a single label problem.

```
$ martingales -h

Test exchangeability using martingales.

Usage: martingales plugin [--bandwidth=<bw>] [options] <output-file> <pvalues-file>
       martingales power [--epsilon=<e>] [options] <output-file> <pvalues-file>
       martingales (--help | --version)

Options:
    --seed                      PRNG seed.
    -h, --help                  Show help.
    --version                   Show the version.
```


# Library

To exploit all the functionalities, or to integrate it into your project,
you may want to use the actual library.

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

# Features

Methods:
- [x] Deterministic and smoothed Conformal Predictors (aka, transductive CP)
- [x] Deterministic Inductive Conformal Predictors (ICP)
- [x] Plug-in and Power martingales for exchangeability testing
- [ ] Venn Predictors

Nonconformity measures:
- [x] k-NN
- [ ] KDE
- [ ] Generic wrapper around ML scorers from existing libraries (e.g., rusty-machine)

Binaries:
- [x] CP (both batch prediction and on-line)
- [x] Martingales
- [x] Inductive CP (batch prediction only)

Bindings:
- [ ] Python bindings

# Authors

* Giovanni Cherubin ([giocher.com](https://giocher.com))

# Similar Projects

- [nonconformist](https://github.com/donlnz/nonconformist/) is a
  Python implementation of CP and ICP.
