pub type WindowSize = u8;

pub const WS: WindowSize = 25;
pub const WL: WindowSize = WS;
pub const WH: WindowSize = WS;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;

// The number of negative training images to start with
pub const NUM_NEG: usize = 20000;

// CONSTANTS HOLDING PATHS/DIRECTORIES
/// Path to images of the object
pub const OBJECT_DIR: &str = "images/training/object";

/// Path to images that are not of object
pub const OTHER_DIR: &str = "images/training/other";

/// Path to images not containing object to slice
pub const SLICE_DIR: &str = "images/training/to_slice";

/// Path to cached training images
pub const CACHED_IMAGES: &str = "cache/images.json";

// CONSTANTS USED IN BUILDING THE CASCADE
/// Path to output the cascade
pub const CASCADE: &str = "cache/cascade.json";

/// The number of strong classifiers in the cascade (used when building from
/// layout)
pub const CASCADE_SIZE: usize = 4;

/// Maximum acceptable false positive rate per layer
pub const MAX_FALSE_POS: f64 = 0.9;

/// The minimum detection rate for each layer of the cascade
pub const MIN_DETECT_RATE: f64 = 0.7;

/// Target overall false positive rate for the cascade
pub const TARGET_FALSE_POS: f64 = 0.99;
