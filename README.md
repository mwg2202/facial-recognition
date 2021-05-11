# object-detection
### An object detection application built using concepts from the Viola-Jones object detection Algorithm

### Training the object detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Optionally change the widow size used in training and detection by editing the constant WS defined in src/primitives.rs (larger values of WS means more features which means more memory use and a longer training time)
  3. Place training images into the directories listed in src/constants.rs
  4. Run `cargo run --release -- process_images` to process training images. The processed data will be outputted to the cache directory.
  5. Run `cargo run --release -- cascade` to build the cascade. The cascade data will be outputted to the output directory.

### Using the object-detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Run `cargo run --release -- detect /path/to/img.png` in the root directoy of the repository where /path/to/img.png can be any path to an image of any name with any common format (png, jpeg, etc.)
  3. A copy of the input image will be outputted with a red rectangle around found instances of the object will be created in the output directory with the same name as the original.
