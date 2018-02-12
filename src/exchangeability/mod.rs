//! Defines Exchangeability Martingales.
//!
//! Exchangeability martingales are tools for testing the
//! exchangeability (and i.i.d.) assumption.
//!
//! # Examples
//!
//! NOTE: this is just a sketch example which *won't* run.
//! ```
//! cp = CP();
//! m = Martingale();
//! 
//! for i in 1..n {
//!     p = cp.predict(x[i]);
//!     M = m.next(p);
//!     cp.update(x[i], y[i]);
//! }
//! ```
pub mod martingales;
