use std::fs;

use image::{
    imageops::{crop_imm, FilterType},
    io::Reader as ImageReader,
    ImageBuffer, Luma, Rgb, RgbImage,
};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::{new_bar, Rectangle, Window, WH_32, WL_32};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
    width: usize,
    height: usize,
}
impl IntegralImage {
    /// Creates an integral image from an image
    pub fn new(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> IntegralImage {
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
        IntegralImage {
            pixels,
            width: w,
            height: h,
        }
    }

    pub fn from_slice_dir(slice_dir: &str) -> Vec<IntegralImage> {
        let mut sliced = Vec::<IntegralImage>::new();
        for img in fs::read_dir(slice_dir).unwrap() {
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .expect("Cannot decode image. (Check for Zone.Identifier)")
                .into_luma8();
            let w = img.width();
            let h = img.height();

            for x in 0..(w / WL_32) {
                for y in 0..(h / WH_32) {
                    let img = crop_imm(&img, x * WL_32, y * WL_32, WL_32, WH_32)
                        .to_image();
                    let image = IntegralImage::new(&img);
                    sliced.push(image);
                }
            }
        }
        sliced
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
    pub fn from_dirs(
        object_dir: &str,
        other_dir: &str,
        slice_dir: &str,
        num_pos: usize,
        num_neg: usize,
    ) -> Vec<ImageData> {
        // Slice images
        let sliced = IntegralImage::from_slice_dir(slice_dir);
        let sliced_size = num_neg - fs::read_dir(other_dir).unwrap().count();
        let sliced = sliced.choose_multiple(&mut rand::thread_rng(), sliced_size);

        // Find the number of objects and others
        let num_objects = fs::read_dir(object_dir).unwrap().count();

        // Create a vector to hold the image data
        let mut set = Vec::<ImageData>::with_capacity(num_objects + num_neg);
        let bar = new_bar(num_objects + num_neg, "Processing Images...");

        // Calculate the weight of each object image
        let weight = 1.0 / (2 * num_pos) as f64;

        // Add each image from the objects directory to the vector
        for img in fs::read_dir(object_dir).unwrap() {
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

            // Push to vector
            set.push(ImageData {
                image,
                weight,
                is_object: true,
            });
            bar.inc(1);
        }
        let mut set: Vec<_> = set
            .choose_multiple(&mut rand::thread_rng(), num_pos)
            .cloned()
            .collect();

        // Calculate the weight of each object image
        let weight = 1.0 / (2 * num_neg) as f64;

        // Add each image from the objects directory to the vector
        for img in fs::read_dir(other_dir).unwrap() {
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

            // Push to vector
            set.push(ImageData {
                image,
                weight,
                is_object: false,
            });
            bar.inc(1);
        }
        for image in sliced.cloned() {
            set.push(ImageData {
                image,
                weight,
                is_object: false,
            });
            bar.inc(1);
        }
        bar.finish();
        set
    }

    /// Normalize the weights of a set of image data
    pub fn normalize_weights(set: &mut [ImageData]) {
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
