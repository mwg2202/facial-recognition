// WINDOW SIZE USED IN TRAINING
pub type WindowSize = u8;
pub const WS: WindowSize = 25;
pub const WL: WindowSize = WS;
pub const WH: WindowSize = WS;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;

// TRAINING DATA SIZE
/// The number of negative training images to start with
pub const NUM_NEG: usize = 10000;

/// The number of positive training images to train over
pub const NUM_POS: usize = 5000;

/// The minimum number of negative training images to be used
pub const MIN_NUM_NEG: usize = 8000;

// CONSTANTS HOLDING PATHS/DIRECTORIES
/// Path to images of the object
pub const OBJECT_DIR: &str = "images/training/object";

/// Path to images that are not of object
pub const OTHER_DIR: &str = "images/training/other";

/// Path to cached training images
pub const CACHED_IMAGES: &str = "cache/images.json";

/// Path to output the cascade
pub const CASCADE: &str = "output/cascade.json";

/// Path to the cascade backup
pub const CASCADE_BACKUP: &str = "cache/cascade_backup.json";

/// Path to backup the current strong classifier
pub const STRONG_CLASSIFIER: &str = "cache/strong_classifier.json";

/// Path to output the found instances of the object in detection
pub const DETECTION_OUTPUT: &str = "output/object.json";

// CONSTANTS USED IN BUILDING THE CASCADE
/// The number of strong classifiers in the cascade (used when building from
/// layout)
pub const CASCADE_SIZE: usize = 4;

/// Maximum acceptable false positive rate per layer
pub const MAX_FALSE_POS: f64 = 0.4;

/// The minimum detection rate for each layer of the cascade
pub const MIN_DETECT_RATE: f64 = 0.9;

/// Target overall false positive rate for the cascade
pub const TARGET_FALSE_POS: f64 = 5e-6;

// In order to find the maximum number of layers that the cascade can have after
// being built using a target false positive rate, by evaluating the log base
// MIN_FALSE_POS of TARGET_FALSE_POS.
// The minimum detection rate of the built cascade is given by evaluating
// MIN_DETECTION_RATE raised to the size of the cascade found in the previous
// calculation.
// The minimum true positive rate is then given by subtracting the maximum false
// positive rate from the minimum detection rate of the cascade.
