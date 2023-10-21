import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor

def process_image(src_file_path, dest_file_path, new_size):
    try:
        with Image.open(src_file_path) as img:
            img = img.resize(new_size, Image.NEAREST)
            img.save(dest_file_path)
        return src_file_path
    except Exception as e:
        return f"Error processing {src_file_path}: {e}"

def resize_and_copy_images(src_dir, dest_dir, new_size):
    with ThreadPoolExecutor(max_workers=24) as executor:  # Adjust the number of workers as needed
        future_to_src_file = {}
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(dest_dir, os.path.relpath(src_file_path, src_dir))
                    dest_file_dir = os.path.dirname(dest_file_path)
                    dir_type = 'label' if 'gtFine' in src_file_path else 'image' if 'leftImg8bit' in src_file_path else 'na'
                    if dir_type == 'label': 
                        dest_file_path = dest_file_path.replace('frame', '')
                    elif dir_type == 'image':
                        dest_file_path = dest_file_path.replace('frame','').replace('_leftImg8bit', '_image')
                    else:
                        Exception(f"The path doesn't seem to be segmentation map or image. {src_file_path} -> {dest_file_path}")
                    os.makedirs(dest_file_dir, exist_ok=True)
                    future = executor.submit(process_image, src_file_path, dest_file_path, new_size)
                    future_to_src_file[future] = src_file_path

        # Process and track results
        for future in tqdm(future_to_src_file):
            src_file_path = future_to_src_file[future]
            result = future.result()
            if result == src_file_path:
                pass
            else:
                print(result)


if __name__ == "__main__":
    source_directory = "/data/divy/idd20k_lite/full_dataset/idd20kII"  # Replace with the source directory path
    destination_directory = "/data/divy/idd20k_lite/full_dataset/small_idd20kII"  # Replace with the destination directory path
    target_resolution = (320, 227)  # Set your desired image resolution

    resize_and_copy_images(source_directory, destination_directory, target_resolution)
