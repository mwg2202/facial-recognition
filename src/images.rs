use std::fs;

use image::{
    imageops::{crop_imm, FilterType},
    io::Reader as ImageReader,
    ImageBuffer, Luma, Rgb, RgbImage,
};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
    width: usize,
    height: usize,
}
impl IntegralImage {
    /// Creates an integral image from an image
    pub fn new(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Self {
        // Calculate each pixel of the integral image
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut pixels = Vec::<u64>::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mut pixel = u64::from(img.get_pixel(x as u32, y as u32)[0]);
                if y != 0 {
                    pixel += pixels[x + w * (y - 1)];
                }
                if x != 0 {
                    pixel += pixels[(x - 1) + w * y];
                }
                if x != 0 && y != 0 {
                    pixel -= pixels[(x - 1) + w * (y - 1)];
                }
                pixels.push(pixel);
            }
        }
        Self {
            pixels,
            width: w,
            height: h,
        }
    }

    /// Gets the sum of pixels in a rectangular region of the original image
    /// using the images corresponding integral image
    pub fn rect_sum(&self, r: &Window, w: Option<(Rectangle<u32>, f64)>) -> i64 {
        let mut xtl = usize::from(r.top_left[0]);
        let mut ytl = usize::from(r.top_left[1]);
        let mut xbr = usize::from(r.bot_right[0]);
        let mut ybr = usize::from(r.bot_right[1]);

        if let Some(w) = w {
            xtl += w.0.top_left[0] as usize;
            ytl += w.0.top_left[1] as usize;
            xbr += w.0.top_left[0] as usize;
            ybr += w.0.top_left[1] as usize;
        }

        let f = w.map(|v| v.1 as i64).unwrap_or(1);

        (self.pixels[xbr + self.width * ybr] as i64
            - self.pixels[xbr + self.width * ytl] as i64
            - self.pixels[xtl + self.width * ybr] as i64
            + self.pixels[xtl + self.width * ytl] as i64)
            / (f * f)
    }
}

/// A struct-of-arrays representing all of the training images
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ImageData {
    pub image: IntegralImage,
    pub weight: f64,
    pub is_object: bool,
}
impl ImageData {
    /// Create image data from a directories
    pub fn from_dirs(cascade: Option<&Cascade>) -> Vec<Self> {
        // Get the negative training images
        let mut set = Self::from_other_dir(cascade);

        // Get the positive training images
        set.append(&mut Self::from_object_dir());

        // Return the images
        set
    }

    /// Gets the training images
    pub fn from_object_dir() -> Vec<Self> {
        // Create a vector to hold the image data
        let mut set = Vec::<Self>::new();

        // Add each image from the objects directory to the vector
        for img in fs::read_dir(OBJECT_DIR).unwrap() {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();

            // Resize the image and turn it to grayscale
            let img = img
                .resize_to_fill(WL_32, WH_32, FilterType::Triangle)
                .into_luma8();

            // Convert image to Integral Image
            let image = IntegralImage::new(&img);

            // Calculate the weight
            let weight = 1.0 / (2 * NUM_POS) as f64;

            // Push to vector
            set.push(Self {
                image,
                weight,
                is_object: true,
            });
        }

        let set: Vec<_> = set
            .choose_multiple(&mut rand::thread_rng(), NUM_POS)
            .cloned()
            .collect();
        set
    }

    /// Slices the images from the other directory and moves them to a vector
    /// of integral images
    pub fn from_other_dir(cascade: Option<&Cascade>) -> Vec<Self> {
        // Create a vector to hold the image data
        let mut set = Vec::<Self>::new();

        // Slice each image in the slice directory and add the slice to the vector
        for img in fs::read_dir(OTHER_DIR).unwrap() {
            // Open the image and convert to greyscale
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .expect("Cannot decode image. (Check for Zone.Identifier)")
                .into_luma8();

            // Calculate the starting weight of each image
            let weight = 1.0 / (2 * NUM_NEG) as f64;

            // Get image height and width
            let w = img.width();
            let h = img.height();

            // Get all possible slices of size WL by WH
            for x in 0..(w / WL_32) {
                for y in 0..(h / WH_32) {
                    // Crop image
                    let img = crop_imm(&img, x, y, WL_32, WH_32).to_image();

                    // Convert image to integral image
                    let image = IntegralImage::new(&img);

                    // Push the image onto the vector
                    if let Some(cascade) = cascade {
                        if cascade.classify(&image, None) {
                            set.push(Self {
                                image,
                                weight,
                                is_object: false,
                            });
                        }
                    } else {
                        set.push(Self {
                            image,
                            weight,
                            is_object: false,
                        });
                    }
                }
            }
        }
        let set: Vec<_> = set
            .choose_multiple(&mut rand::thread_rng(), NUM_NEG)
            .cloned()
            .collect();
        set
    }

    /// Normalize the weights of a set of image data
    pub fn normalize_weights(set: &mut [Self]) {
        // Sum over the weights of all the images
        let sum: f64 = set.iter().map(|d| d.weight).sum();

        // Divide each image's original weight by the sum
        for data in set.iter_mut() {
            data.weight /= sum;
        }
    }
}

/// Draws a rectangle over an image
pub fn draw_rectangle(img: &mut RgbImage, r: &Rectangle<u32>) {
    let pixel: Rgb<u8> = Rgb::from([0x88, 0x95, 0x8D]);
    for x in r.top_left[0]..r.bot_right[0] {
        img.put_pixel(x, r.top_left[1], pixel);
        img.put_pixel(x, r.bot_right[1], pixel);
    }
    for y in r.top_left[1]..r.bot_right[1] {
        img.put_pixel(r.top_left[0], y, pixel);
        img.put_pixel(r.bot_right[0], y, pixel);
    }
}
