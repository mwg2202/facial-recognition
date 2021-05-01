use serde::{Deserialize, Serialize};

use super::{ImageData, IntegralImage, Rectangle, WeakClassifier};

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Debug, Serialize, Deserialize)]
pub struct StrongClassifier {
    pub wcs: Vec<WeakClassifier>,
    weights: Vec<f64>,
}
impl StrongClassifier {
    /// Creates an empty strong classifier
    pub fn new() -> StrongClassifier {
        // Build and return the strong classifier
        let wcs = Vec::<WeakClassifier>::new();
        let weights = Vec::<f64>::new();
        StrongClassifier { wcs, weights}
    }

    /// Creates a weak classifier which is then pushed onto self
    pub fn push(&mut self, wcs: &mut[WeakClassifier], set: &mut[ImageData]) {
        // Normalize weights
        ImageData::normalize_weights(set);

        // Calculate Thresholds
        WeakClassifier::calculate_thresholds(wcs, set);

        // Get the best weak classifier based off of the training images
        let wc = WeakClassifier::get_best(wcs, set);

        // Update the weights
        let weight = wc.update_weights(set);
        self.wcs.push(wc); self.weights.push(weight);
    }

    /// Classifies a rectangular portion of an integral image as being a face or not
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
            >= self.weights.iter().sum::<f64>() / 2.0
    }
}
