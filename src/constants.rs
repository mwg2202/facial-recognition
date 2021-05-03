// WINDOW SIZE USED IN TRAINING
pub type WindowSize = u8;
pub const WS: WindowSize = 25;
pub const WL: WindowSize = WS;
pub const WH: WindowSize = WS;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;
pub const STEP_SIZE: usize = 5;

// TRAINING DATA SIZE
/// The number of negative training images to start with
pub const NUM_NEG: usize = 5000;

/// The number of positive training images to train over
pub const NUM_POS: usize = 5000;

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
/// The number of strong classifiers in the cascade
pub const CASCADE_SIZE: usize = 10;

/// Maximum acceptable false positive rate per layer
pub const MAX_FALSE_POS: f64 = 0.4;

/// The minimum detection rate for each layer of the cascade
pub const MIN_DETECT_RATE: f64 = 0.95;

// The maximum fianl false positive positive rate is given MAX_FALSE_POS raised
// to CASCADE_SIZE
// The minimum detection rate of the built cascade is given by evaluating
// MIN_DETECTION_RATE raised to CASCADE_SIZE
// The minimum true positive rate is then given by subtracting the maximum false
// positive rate from the minimum detection rate of the cascade.
