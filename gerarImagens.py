import os
from PIL import Image

# Function to resize and save image
def resize_and_save(image, size, filename):
    # Resize the image using high-quality Lanczos interpolation
    resized_image = image.resize(size, Image.LANCZOS)
    # Save the resized image
    resized_image.save(filename)
    print(f"Saved {filename}")

# Main function
def main():
    base_filename = 'base.png'
    
    # Check if base image exists
    if not os.path.exists(base_filename):
        print(f"Error: {base_filename} not found.")
        return
    
    resize_and_save(Image.open(base_filename), (512, 512), 'image_512x512.png')
    resize_and_save(Image.open(base_filename), (1024, 1024), 'image_1024x1024.png')
    resize_and_save(Image.open(base_filename), (2048, 2048), 'image_2048x2048.png')
    resize_and_save(Image.open(base_filename), (4096, 4096), 'image_4096x4096.png')

if __name__ == "__main__":
    main()