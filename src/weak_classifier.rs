use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    new_bar, Feature, ImageData, IntegralImage, OrderedF64, Rectangle, Window, WH,
    WL,
};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct WeakClassifier {
    feature: Feature,
    pub threshold: i64,
    pub pos_polarity: bool,
}
impl WeakClassifier {
    pub const fn new(feature: Feature) -> Self {
        Self {
            feature,
            threshold: 0,
            pos_polarity: false,
        }
    }

    pub fn filter(mut wcs: Vec<Self>, set: &mut [ImageData]) -> Vec<Self> {
        Self::calculate_thresholds(&mut wcs, set);
        wcs.sort_by_cached_key(|wc: &Self| OrderedF64(wc.error(set)));
        println!("Sorted");
        wcs.reverse();
        println!("Reversed");
        wcs.truncate(wcs.len() / 10);
        wcs
    }

    /// Calculates the optimal threshold and polarity for the weak classifier
    pub fn calculate_threshold(&mut self, set: &[ImageData], afs: f64, abg: f64) {
        // Sort the training images based on it's evaluation
        let mut sorted: Vec<_> = set.iter().collect();
        sorted.sort_unstable_by(|a: &&ImageData, b: &&ImageData| {
            let a_eval = self.feature.evaluate(&a.image, None);
            let b_eval = self.feature.evaluate(&b.image, None);
            a_eval.cmp(&b_eval)
        });

        // Set up variables used in the following loop
        let mut cf: usize = 0; // Total number of pos samples seen so far
        let mut cg: usize = 0; // Total number of neg samples seen so far
        let mut fs: f64 = 0.0; // Sum of the weights of the pos samples seen so far
        let mut bg: f64 = 0.0; // Sum of the weights of neg samples seen so far
        let mut min_err: f64 = 1.0; // The minimum value of the error function
        let mut best_image = &set[0].image;

        for data in set.iter() {
            // Add the weight to fs/bg
            if data.is_object {
                fs += data.weight;
                cf += 1;
            } else {
                bg += data.weight;
                cg += 1;
            }

            // Compute the error function
            let err = f64::min(bg + afs - fs, fs + abg - bg);

            // If we found a threshold with less error update
            // min_err and the polarity
            if err < min_err {
                min_err = err;
                best_image = &data.image;
                self.pos_polarity = cf > cg;
            }
        }
        self.threshold = self.feature.evaluate(best_image, None);
    }

    /// Calculate the optimal thresholds for a slice of weak classifiers.
    /// Calls calculate_threshold() on multiple threads
    pub fn calculate_thresholds(wcs: &mut [Self], set: &[ImageData]) {
        // Calculate the optimal thresholds for all weak classifiers
        let afs = set
            .iter()
            .filter(|data| data.is_object)
            .map(|data| data.weight)
            .sum();
        let abg = set
            .iter()
            .filter(|data| !data.is_object)
            .map(|data| data.weight)
            .sum();
        let bar = new_bar(wcs.len() as u64, "Calculating Thresholds...");
        wcs.par_iter_mut().for_each(|wc| {
            wc.calculate_threshold(set, afs, abg);
            bar.inc(1);
        });
        bar.finish();
    }

    /// Gets all possible weak classifiers
    pub fn get_all() -> Vec<Self> {
        let mut wcs = Vec::<Self>::with_capacity(200_000);
        for w in 1..(WL + 1) {
            for h in 1..(WH + 1) {
                let mut i = 0;
                while (i + w) < WL {
                    let mut j = 0;
                    while (j + h) < WH {
                        // Horizontal Two Rectangle Features
                        if (i + 2 * w) < WL {
                            let white = (Window::new(i, j, w, h), None);
                            let black = (Window::new(i + w, j, w, h), None);
                            let wc = Self::new(Feature { white, black });
                            wcs.push(wc);
                        }

                        // Vertical Two Rectangle Feature
                        if (j + 2 * h) < WH {
                            let white = (Window::new(i, j, w, h), None);
                            let black = (Window::new(i, j + h, w, h), None);
                            let wc = Self::new(Feature { white, black });
                            wcs.push(wc);
                        }

                        // Horizontal Three Rectangle Feature
                        if (i + 3 * w) < WL {
                            let white = (
                                Window::new(i, j, w, h),
                                Some(Window::new(i + 2 * w, j, w, h)),
                            );
                            let black = (Window::new(i + w, j, w, h), None);
                            let wc = Self::new(Feature { white, black });
                            wcs.push(wc);
                        }

                        // Vertical Three Rectangle Feature
                        if (j + 3 * h) < WH {
                            let white = (
                                Window::new(i, j, w, h),
                                Some(Window::new(i, j + 2 * h, w, h)),
                            );
                            let black = (Window::new(i, j + h, w, h), None);
                            let wc = Self::new(Feature { white, black });
                            wcs.push(wc);
                        }

                        // Four rectangle features
                        if (i + 2 * w) < WL && (j + 2 * h) < WH {
                            let white = (
                                Window::new(i, j, w, h),
                                Some(Window::new(i + w, j + h, w, h)),
                            );
                            let black = (
                                Window::new(i + w, j, w, h),
                                Some(Window::new(i, j + h, w, h)),
                            );
                            let wc = Self::new(Feature { white, black });
                            wcs.push(wc);
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }
        }
        wcs
    }

    // Calculates the error of a wc over a given training set
    pub fn error(&self, set: &[ImageData]) -> f64 {
        set.iter()
            .filter(|data| data.is_object != self.classify(&data.image, None))
            .map(|data| data.weight)
            .sum::<f64>()
    }

    // Gets the best weak classifier over a given training set
    pub fn get_best(wcs: &[Self], set: &[ImageData]) -> Self {
        // Find the best weak classifier
        *wcs.iter()
            .min_by_key(|wc| OrderedF64(wc.error(set)))
            .expect("wcs was empty")
    }

    // Updates the weights of the images based off of the error of the
    // self over the images
    pub fn update_weights(&self, set: &mut [ImageData]) -> f64 {
        let err = self.error(set);
        let beta_t = err / (1.0 - err);

        // Update the weights:
        set.iter_mut()
            .filter(|data| self.classify(&data.image, None) == data.is_object)
            .for_each(|data| {
                data.weight *= beta_t;
            });
        f64::ln(1.0 / beta_t)
    }

    pub fn classify(
        &self,
        ii: &IntegralImage,
        w: Option<(Rectangle<u32>, f64)>,
    ) -> bool {
        self.pos_polarity == (self.feature.evaluate(ii, w) < self.threshold)
    }
}
