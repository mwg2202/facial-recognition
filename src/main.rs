mod cascade;
mod constants;
mod images;
mod primitives;
mod strong_classifier;
mod weak_classifier;

use std::path::Path;

pub use cascade::Cascade;
use clap::{load_yaml, App, AppSettings};
pub use constants::*;
use image::io::Reader as ImageReader;
pub use images::{draw_rectangle, ImageData, IntegralImage};
pub use primitives::*;
pub use strong_classifier::StrongClassifier;
pub use weak_classifier::WeakClassifier;

fn main() {
    // Parse the cli arguments using clap
    let yaml = load_yaml!("cli.yml");
    let app = App::from_yaml(yaml)
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // Run the specified subcommand
    match app.subcommand() {
        ("process_images", Some(_)) => process_images(),
        ("cascade", Some(_)) => cascade(),
        ("test", Some(_)) => test(),
        ("detect", Some(m)) => detect(m),
        ("continue", Some(_)) => continue_training(),
        _ => println!("Incorrect subcommand"),
    }
}

/// Processes images for use in building the cascade
fn process_images() {
    // Find and process images
    println!("Processing training images:");
    let set = ImageData::from_dirs(None);
    println!("Processed {} images.", set.len());

    // Save image data to cache
    let data = serde_json::to_string(&set).unwrap();
    std::fs::write(CACHED_IMAGES, &data).expect("Unable to cache images");
}

/// Builds the cascade
fn cascade() {
    // Get training images from cache or process raw images if cache is empty
    let set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached images")
        } else {
            println!("Training image data not found in cache");
            return;
        }
    };

    // Get all possible weak classifiers
    let mut wcs = WeakClassifier::get_all();
    println!("{:-^30}", " Getting Weak Classifiers ");
    println!("Created vector of {} possible weak classifiers", wcs.len());
    // println!("Obatining the top 10% of weak classifiers");
    // let mut wcs = WeakClassifier::filter(wcs, &mut set);

    println!("{:-^30}", " Building Cascade ");

    // Build the cascade using the weak classifiers
    let cascade = Cascade::from_false_pos(&mut wcs, set);

    // Output the data
    println!("Saving cascade to {}", CASCADE);
    let data = serde_json::to_string_pretty(&cascade).unwrap();
    std::fs::write(CASCADE, &data).expect("Unable to write to file");
}

/// Test a cascade over training images
fn test() {
    // Get the cached cascade
    let cascade: Cascade = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached cascade")
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };

    // Get processed training images from cache
    let set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached image data")
        } else {
            println!("Testing image data not found in cache");
            return;
        }
    };

    // Test the cascade over training images
    println!("Testing the Cascade");
    let mut correct_objects: f64 = 0.0;
    let mut correct_others: f64 = 0.0;
    let mut num_objects: f64 = 0.0;
    for data in set.iter() {
        let eval = cascade.classify(&data.image, None);
        if data.is_object {
            num_objects += 1.0;
        }
        if data.is_object && eval {
            correct_objects += 1.0;
        }
        if !data.is_object && !eval {
            correct_others += 1.0;
        }
    }

    // Print test results
    println!("correct_objects: {}", correct_objects);
    println!("correct_others: {}", correct_others);
    println!("num_objects: {}", num_objects);
    println!("images_len: {}", set.len());
    println!(
        "Percent of correctly evaluated images: {:.2}%",
        (correct_objects + correct_others) * 100.0 / (set.len() as f64)
    );
    println!(
        "Percent of correctly evaluated images of the object: {:.2}%",
        correct_objects * 100.0 / num_objects
    );
    println!(
        "Percent of correctly evaluated images which aren't the object: {:.2}%",
        correct_others * 100.0 / (set.len() as f64 - num_objects)
    );
}

/// This detects objects by sending a "windowed" view into the image to
/// be evaluated by the cascade. The window moves across the image and grows
/// in size. This tests all rectangles in the images for the object
fn detect(m: &clap::ArgMatches) {
    // Get the cached cascade
    let cascade: Cascade = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };

    // Get the input image
    let path = m.value_of("input_image").unwrap();

    // Get the location to store the output image
    let output_img = "output/".to_owned()
        + Path::new(path).file_name().unwrap().to_str().unwrap();

    // Open the image
    let img = ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_luma8();

    // Get the size of the image
    let size = (img.width(), img.height());

    // Convert image to integral image
    let ii = IntegralImage::new(&img);

    // Find all instances of the object in the image
    let rects = cascade.detect(&ii, size);

    // Edit the original image to show the found instances
    let mut img = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
    for r in rects.iter() {
        draw_rectangle(&mut img, r);
    }

    // Output the editied image
    img.save(output_img).unwrap();

    // Output the rectangles
    let data = serde_json::to_string_pretty(&rects).unwrap();
    std::fs::write(DETECTION_OUTPUT, &data).expect("Unable to write to file");
}
pub fn continue_training() {
    // Get the current cascade
    let mut cascade: Cascade = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };

    // Get processed training images from cache
    let mut set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached image data")
        } else {
            println!("Testing image data not found in cache");
            return;
        }
    };

    // Keep the positive training samples
    set.retain(|data| data.is_object);

    // Recalculate the negative training samples
    set.append(&mut ImageData::from_other_dir(Some(&cascade)));

    // Get all possible weak classifiers
    let mut wcs = WeakClassifier::get_all();

    // Train the cascade
    cascade.train(&mut wcs, set);
}
