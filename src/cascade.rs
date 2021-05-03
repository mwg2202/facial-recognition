use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct Cascade(Vec<StrongClassifier>);
impl Cascade {
    fn new() -> Cascade { Cascade(Vec::<StrongClassifier>::new()) }

    pub fn classify(
        &self,
        ii: &IntegralImage,
        r: Option<(Rectangle<u32>, f64)>,
    ) -> bool {
        self.0.iter().all(|sc| sc.classify(&ii, r))
    }

    pub fn from_false_pos(
        wcs: &mut [WeakClassifier],
        set: Vec<ImageData>,
    ) -> Cascade {
        // Create an empty cascade
        let mut cascade = Cascade::new();
        cascade.train(wcs, set);
        cascade
    }

    pub fn train(&mut self, wcs: &mut [WeakClassifier], mut set: Vec<ImageData>) {
        // Calculates the detection rate and false positive rate
        // of a layer of the cascade
        let detect_rate = |sc: &StrongClassifier, set: &[ImageData]| -> f64 {
            let results = set
                .iter()
                .map(|data| (data.is_object, sc.classify(&data.image, None)));
            let num_pos = results.clone().filter(|(a, _)| *a).count() as f64;
            results.filter(|(a, b)| *a && *b).count() as f64 / num_pos
        };

        let false_pos = |sc: &StrongClassifier, set: &[ImageData]| -> f64 {
            let results = set
                .iter()
                .map(|data| (data.is_object, sc.classify(&data.image, None)));
            let num_neg = results.clone().filter(|(a, _)| !*a).count() as f64;
            let fp = results.filter(|(a, b)| !*a && *b).count() as f64 / num_neg;
            println!("Current False Positive Rate (Layer): {}", fp);
            fp
        };

        // Build the cascade
        for _ in 0..CASCADE_SIZE {
            let mut sc = StrongClassifier::new();
            println!("Building Strong Classifier #{}", self.0.len() + 1);
            loop {
                // Add a weak classifier to the strong classifier
                println!("Building Weak Classifier #{}", sc.wcs.len() + 1);
                sc.push(wcs, &mut set);

                // If not enough images are being detected, increase the
                // threshold of the current weak classifier
                println!("Resetting Strong Classifier Threshold");
                let step = sc.threshold / 300.0;
                while detect_rate(&sc, &set) < MIN_DETECT_RATE {
                    let min_weight =
                        *sc.weights.iter().min_by_key(|&w| OrderedF64(*w)).unwrap();

                    // If the threshold is less than the minimum weight,
                    // then decreasing it to any value other than 0 will have
                    // no affect, so we go ahead and set it to 0
                    if sc.threshold < min_weight {
                        sc.threshold = 0.0
                    } else {
                        sc.threshold -= step
                    }
                }
                println!("Current Detect Rate (Layer): {}", detect_rate(&sc, &set));
                println!("New Threshold: {}", sc.wcs.last().unwrap().threshold);

                // Save current strong classifier
                let data = serde_json::to_string_pretty(&sc).unwrap();
                std::fs::write(STRONG_CLASSIFIER, &data)
                    .expect("Unable to write strong classifier to file");

                if false_pos(&sc, &set) < MAX_FALSE_POS {
                    break;
                }
            }

            // Add the strong classifier to the cascade
            self.0.push(sc);

            // Recalculate the negative training samples
            set = ImageData::from_dirs(Some(self));

            // Backup the cascade
            let data = serde_json::to_string_pretty(self).unwrap();
            std::fs::write(CASCADE_BACKUP, &data)
                .expect("Unable to backup cascade to file");
        }
    }

    /// Detects all instances of the object in an image from it's integral image
    pub fn detect(
        &self,
        img: &IntegralImage,
        size: (u32, u32),
    ) -> Vec<Rectangle<u32>> {
        // Vector to hold detected objects
        let mut objects = Vec::<Rectangle<u32>>::new();

        // Determine what is the largest window size that can fit in the image
        let max_w = if (size.0 / WL_32) < (size.1 / WH_32) {
            size.0
        } else {
            size.1 * WL_32 / WH_32
        };

        // Search the image for the object
        for w in (WL_32..=max_w).step_by(STEP_SIZE) {
            let h = w * WH_32 / WL_32;
            let f = f64::from(w) / f64::from(WL_32);
            for x in 0..(size.0 - w) {
                for y in 0..(size.1 - h) {
                    let r = Rectangle::<u32>::new(x, y, w, h);
                    if self.classify(img, Some((r, f))) {
                        objects.push(r);
                    }
                }
            }
        }
        objects
    }
}
