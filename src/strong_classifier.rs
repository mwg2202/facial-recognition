use serde::{Deserialize, Serialize};

use super::*;

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Debug, Serialize, Deserialize)]
pub struct StrongClassifier {
    pub wcs: Vec<WeakClassifier>,
    pub weights: Vec<f64>,
    pub threshold: f64,
}
impl StrongClassifier {
    /// Creates an empty strong classifier
    pub const fn new() -> Self {
        // Build and return the strong classifier
        let wcs = Vec::<WeakClassifier>::new();
        let weights = Vec::<f64>::new();
        let threshold = 0.0;
        Self {
            wcs,
            weights,
            threshold,
        }
    }

    /// Creates a weak classifier which is then pushed onto self
    pub fn push(&mut self, wcs: &mut [WeakClassifier], set: &mut [ImageData]) {
        // Normalize weights
        ImageData::normalize_weights(set);

        // Calculate Thresholds
        WeakClassifier::calculate_thresholds(wcs, set);

        // Get the best weak classifier based off of the training images
        let wc = WeakClassifier::get_best(wcs, set);

        // Update the weights
        let weight = wc.update_weights(set);
        self.wcs.push(wc);
        self.weights.push(weight);

        // Find the threshold of the strong classifier
        self.threshold = self.weights.iter().sum::<f64>() / 2.0;
    }

    /// Classifies a rectangular portion of an integral image as being a face or
    /// not
    pub fn classify(
        &self,
        ii: &IntegralImage,
        w: Option<(Rectangle<u32>, f64)>,
    ) -> bool {
        self.wcs
            .iter()
            .zip(self.weights.iter())
            .filter(|(wc, _)| wc.classify(ii, w))
            .map(|(_, weight)| weight)
            .sum::<f64>()
            >= self.threshold
    }
}
impl Default for StrongClassifier {
    fn default() -> Self { Self::new() }
}
