//! Module defining Conformal Predictors.
//!
//! A `ConfidencePredictor<T>` implements all methods to provide a confidence
//! prediction for new input vectors.
//! Examples of confidence predictors are inductive and transductive
//! Conformal Predictors.
pub mod cp;

use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

pub use self::cp::CP;


/// A Confidence Predictor (either transductive or inductive)
///
/// This trait is parametrized over `T`, the element type.
/// It provides all the methods for making a confidence prediction.
pub trait ConfidencePredictor<T> {
    /// Trains a Conformal Predictor on a training set.
    ///
    /// Pedantic note: because CP is a transductive method, it never
    /// actually trains a model.
    /// This function, however, structures the training data so that
    /// it can be easily used in the prediction phase.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    ///
    /// # Examples
    ///
    /// Please, see [CP](/cp/cp/struct.CP.html).
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>) -> LearningResult<()>;
    /// Updates a Conformal Predictor with more training data.
    ///
    /// After calling `train()` once, `update()` allows to add
    /// inputs to the Conformal Predictor's training data,
    /// which will be used for future predictions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    ///
    /// # Examples
    ///
    /// Please, see [CP](/cp/cp/struct.CP.html).
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>) -> LearningResult<()>;
    /// Returns candidate labels (region prediction) for test vectors.
    ///
    /// The return value is a matrix of `bool` (`Array2<bool>`) with shape
    /// `(n_inputs, n_labels)`, where `n_inputs = inputs.rows()` and
    /// `n_labels` is the number of possible labels;
    /// in such matrix, each column `y` corresponds to a label,
    /// each row `i` to an input object, and the value at `[i,y]` is
    /// true if the label conforms the distribution, false otherwise.
    ///
    /// # Examples
    ///
    /// Please, see [CP](/cp/cp/struct.CP.html).
    fn predict(&mut self, inputs: &ArrayView2<T>) -> LearningResult<Array2<bool>>;
    /// Returns the p-values for test vectors.
    ///
    /// The return value is a matrix of `f64` (`Array2<f64>`) with shape
    /// `(n_inputs, n_labels)`, where `n_inputs = inputs.rows()` and
    /// `n_labels` is the number of possible labels;
    /// in such matrix, each column `y` corresponds to a label,
    /// each row `i` to an input object, and the value at `[i,y]` is
    /// the p-value obtained when assuming `y` as a label for the
    /// `i`-th input object.
    ///
    /// # Examples
    ///
    /// Please, see [CP](/cp/cp/struct.CP.html).
    fn predict_confidence(&mut self, inputs: &ArrayView2<T>) -> LearningResult<Array2<f64>>;
    /// Sets the significance level.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Significance level in [0,1].
    fn set_epsilon(&mut self, epsilon: f64);
}
