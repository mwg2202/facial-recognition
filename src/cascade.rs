use serde::{Deserialize, Serialize};

use super::{
    ImageData, IntegralImage, Rectangle, StrongClassifier, WeakClassifier,
    CASCADE_BACKUP, STRONG_CLASSIFIER, OrderedF64, WL_32, WH_32, WL,
    MIN_NUM_NEG, OTHER_DIR, NUM_NEG
};

#[derive(Serialize, Deserialize, Debug)]
pub struct Cascade(Vec<StrongClassifier>);
impl Cascade {
    fn new() -> Cascade { Cascade(Vec::<StrongClassifier>::new()) }

    fn with_capacity(size: usize) -> Cascade {
        Cascade(Vec::<StrongClassifier>::with_capacity(size))
    }

    pub fn classify(
        &self,
        ii: &IntegralImage,
        r: Option<(Rectangle<u32>, f64)>,
    ) -> bool {
        self.0.iter().all(|sc| sc.classify(&ii, r))
    }

    pub fn from_layout(
        layout: &Vec<usize>,
        wcs: &mut Vec<WeakClassifier>,
        set: &mut Vec<ImageData>,
    ) -> Cascade {
        // Build the cascade using the weak classifiers
        let cascade_size = layout.len();
        let mut cascade = Cascade::with_capacity(cascade_size);
        layout.iter().enumerate().for_each(|(i, &size)| {
            println!("Building Strong Classifier {} of {}", i + 1, cascade_size);

            // Create the strong classifier
            let mut sc = StrongClassifier::new();
            println!("Building Strong Classifier {} of {}", i, cascade_size);
            (1..=size).for_each(|j| {
                println!("Adding Weak Classifier {} of {}", j, size);
                sc.push(wcs, set);
            });

            // Remove the true negatives from the training set
            set.retain(|data| data.is_object || sc.classify(&data.image, None));

            // Add the strong classifier to the cascade
            cascade.0.push(sc);
        });
        cascade
    }

    pub fn from_false_pos(
        wcs: &mut Vec<WeakClassifier>,
        set: &mut Vec<ImageData>,
        max_false_pos: f64,
        min_detect_rate: f64,
        target_false_pos: f64,
    ) -> Cascade {
        // Create an empty cascade
        let mut cascade = Cascade::new();
        cascade.train(
            wcs, 
            set, 
            max_false_pos, 
            min_detect_rate, 
            target_false_pos,
        );
        cascade
    }

    pub fn train(
        &mut self,
        wcs: &mut Vec<WeakClassifier>,
        set: &mut Vec<ImageData>,
        max_false_pos: f64,
        min_detect_rate: f64,
        target_false_pos: f64,
    ) {
        // Create an empty cascade
        let mut curr_set = set.clone();

        // Calculates the false positive rate for the cascade
        let total_false_pos = |cascade: &Cascade, set: &[ImageData]| 
            -> f64 {
            let results = set
                .iter()
                .map(|data| (data.is_object, cascade.classify(&data.image, None)));
            let num_neg = results.clone().filter(|(a, _)| !*a).count() as f64;
            let tfp = results.filter(|(a, b)| !*a && *b).count() as f64 / num_neg;
            println!("Current False Positive Rate (Cascade): {}", tfp);
            tfp
        };

        // Calculates the detection rate and false positive rate
        // of a layer of the cascade
        let detect_rate = |sc: &StrongClassifier, curr_set: &[ImageData]| -> f64 {
            let results = curr_set
                .iter()
                .map(|data| (data.is_object, sc.classify(&data.image, None)));
            let num_pos = results.clone().filter(|(a, _)| *a).count() as f64;
            results.filter(|(a, b)| *a && *b).count() as f64 / num_pos
        };

        let false_pos = |sc: &StrongClassifier, curr_set: &[ImageData]| -> f64 {
            let results = curr_set
                .iter()
                .map(|data| (data.is_object, sc.classify(&data.image, None)));
            let num_neg = results.clone().filter(|(a, _)| !*a).count() as f64;
            let fp = results.filter(|(a, b)| !*a && *b).count() as f64 / num_neg;
            println!("Current False Positive Rate (Layer): {}", fp);
            fp
        };
        while total_false_pos(self, set) > target_false_pos {
            let mut sc = StrongClassifier::new();
            println!("Building Strong Classifier #{}", self.0.len() + 1);
            loop {
                // Add a weak classifier to the strong classifier
                println!("Building Weak Classifier #{}", sc.wcs.len() + 1);
                sc.push(wcs, &mut curr_set);

                // If not enough images are being detected, increase the
                // threshold of the current weak classifier
                println!("Resetting Strong Classifier Threshold");
                let step = sc.threshold / 300.0;
                while detect_rate(&sc, &curr_set) < min_detect_rate {
                    let min_weight = *sc.weights.iter().min_by_key(|&w| OrderedF64(*w)).unwrap();
                    
                    // If the threshold is less than the minimum weight,
                    // then decreasing it to any value other than 0 will have
                    // no affect, so we go ahead and set it to 0
                    if sc.threshold < min_weight { sc.threshold = 0.0 }
                    else { sc.threshold -= step }
                }
                println!("Current Detect Rate (Layer): {}", 
                    detect_rate(&sc, &curr_set));
                println!("New Threshold: {}", sc.wcs.last().unwrap().threshold);

                // Save current strong classifier
                let data = serde_json::to_string_pretty(&sc).unwrap();
                std::fs::write(STRONG_CLASSIFIER, &data)
                    .expect("Unable to write strong classifier to file");

                if false_pos(&sc, &curr_set) < max_false_pos {
                    break;
                }
            }

            // Remove the true negatives from the training set
            curr_set.retain(|data| data.is_object || sc.classify(&data.image, None));

            if curr_set.iter().filter(|data| !data.is_object).count() < MIN_NUM_NEG {
                // Keep the positive training samples
                set.retain(|data| data.is_object);

                // Recalculate the negative training samples
                set.append(
                    &mut ImageData::from_other_dir(
                        OTHER_DIR, 
                        NUM_NEG, 
                        Some(self)
                    )
                );
            }

            // Add the strong classifier to the cascade
            self.0.push(sc);

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
        let step_size = (f64::from(WL) / 5.0).round() as usize;
        for w in (WL_32..= max_w).step_by(step_size) {
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
