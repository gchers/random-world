use std::f64;
use statrs::statistics::Variance;
/// Exchangeability Martingale.
///
/// A generic exchangeability martingale, as described for example
/// in [1,2].
///
/// [1] "Testing Exchangeability On-Line" (Vovk et al., 2003).
/// [2] "Plug-in martingales for testing exchangeability on-line"
///     (Fedorova et al., 2012).
pub struct Martingale {
    /// Current value of the martingale.
    current: f64,
    /// Threshold to determine if the martingale is "large".
    pub threshold: f64,
    /// Some methods need to record previous p-values.
    pvalues: Option<Vec<f64>>,
    /// The martingale M is updated given a new p-value p
    /// as:
    ///     M *= update_function(p, pvalues)
    /// where pvalues are optionally recorded previous p-values.
    update_function: Box<Fn(f64, &Option<Vec<f64>>) -> f64>,
}

impl Default for Martingale {
    fn default() -> Martingale {
        Martingale {
            current: 1.0,
            threshold: 100.0,
            pvalues: None,
            // Placeholder update_function.
            update_function: Box::new(|_, _| { f64::NAN }),
        }
    }
}


impl Martingale {
    /// Creates a new Power martingale.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Parameter of the Power martingale.
    fn new_power(epsilon: f64) -> Martingale {
        assert!(epsilon >= 0.0 && epsilon <= 1.0);

        Martingale {
            update_function: Box::new(move |pvalue, _| {
                                        epsilon*pvalue.powf(epsilon-1.0)
                                    }),
            pvalues: Some(vec![]),
            ..Default::default()
        }
    }

    /// Creates a new Simple Mixture martingale.
    fn new_simple_mixture() -> Martingale {
        unimplemented!();
    }

    /// Creates a new Plug-in martingale.
    ///
    /// To estimate the density it uses KDE with a gaussian kernel.
    /// If bandwidth is not specified, it uses Silverman's rule of thumb
    /// to determine its value.
    ///
    /// # Arguments
    ///
    /// * `bandwidth` - Bandwidth for the gaussian kernel in KDE.
    fn new_plugin(bandwidth: Option<f64>) -> Martingale {
        Martingale {
            update_function: Box::new(move |pvalue, pvalues| {
                                       kde(pvalue,
                                           &pvalues.as_ref()
                                                   .expect("Plug-in martingale badly initialized"),
                                           bandwidth)
                                     }),
            ..Default::default()
        }
    }

    /// Creates a new martingale from a custom update function.
    ///
    /// # Arguments
    ///
    /// * `update_function` - Function used to update the martingale;
    ///     such function should take as input a p-value and (optionally)
    ///     the previous p-values, and return the resulting value.
    /// * `store_pvalues` - Whether `update_function` requires knowing
    ///     the previous p-values. If set to `false`, `self.pvalues`
    ///     is set to `None`.
    fn from_function(update_function: Box<Fn(f64, &Option<Vec<f64>>) -> f64>,
            store_pvalues: bool) -> Martingale {
        let pvalues = match store_pvalues {
            true => Some(vec![]),
            false => None,
        };

        Martingale {
            update_function: update_function,
            pvalues: pvalues,
            ..Default::default()
        }
    }

    /// Returns the current value of the martingale.
    fn current(&self) -> f64 {
        self.current
    }

    /// Updates the martingale and returns its new value.
    ///
    /// # Arguments
    ///
    /// * `pvalue` - The new observed p-value.
    fn next(&mut self, pvalue: f64) -> f64 {
        // Update.
        self.current = (self.update_function)(pvalue, &self.pvalues);
        // Store if required by the method.
        if let Some(pvalues) = self.pvalues.as_mut() {
            pvalues.push(pvalue);
        }

        self.current
    }

    /// True if the current value of the martingale is larger
    /// than the selected threshold.
    fn is_large(&self) -> bool {
        self.current > self.threshold
    }
}

/// Computes Kernel Density Estimate of a new observation given
/// previous ones.
///
/// It uses a gaussian kernel.
/// If bandwidth is not specified, it uses Silverman's rule of thumb
/// to determine its value.
///
/// # Arguments
///
/// * `x` - New observation.
/// * `x_previous` - Previous observations.
/// * `bandwidth` - Bandwidth for the gaussian kernel in KDE.
fn kde(x: f64, x_previous: &Vec<f64>, bandwidth: Option<f64>) -> f64 {

    let n = x_previous.len() as f64;

    let h = match bandwidth {
        Some(h) => h,
        None => { // Silverman's rule of thumb.
                  x_previous.std_dev()*(4./3./n).powf(1./5.)
                },
    };
    
    // TODO: maybe hardcode value?
    let q = (2.0*f64::consts::PI).sqrt();

    x_previous.iter()
              .map(|xi| (x - xi) / h)
              .map(|u| (-0.5*u.powi(2).exp()))
              .sum::<f64>() / (n*h*q)
}
