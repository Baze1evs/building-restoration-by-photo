import os
import numpy as np
from keras.utils import load_img, img_to_array, array_to_img
import time

image_size = (256, 256)
noise_size = (64, 64)
num_channels = 3
base_path = "C:\\Users\\Bazelevs\\Downloads\\arcDataset"
stop_entries = ["Achaemenid architecture", "Ancient Egyptian architecture", "Novelty architecture"]
skipped = 0


def process_img(img, filename):
    array = img_to_array(img)

    for i in range(image_size[0] // noise_size[0]):
        for j in range(image_size[1] // noise_size[1]):
            norm_array = array - array.mean()
            norm_array /= norm_array.std()
            for k in range(num_channels):
                noise = np.random.normal(0, 1, noise_size)
                norm_array[i * noise_size[0]:(i + 1) * noise_size[0],
                j * noise_size[1]:(j + 1) * noise_size[1],
                k] = noise

            new_img = array_to_img(norm_array)

            save_path = os.path.join("MyDataset", filename.split(".")[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            new_img.save(os.path.join(save_path, f"{i*4 + j}.jpg"))


start_time = time.time()
for entry in os.listdir(base_path):
    path = os.path.join(base_path, entry)
    if os.path.isdir(path) and entry not in stop_entries:
        for filename in os.listdir(path):
            path_to_img = os.path.join(path, filename)
            if "%" in filename:
                path_to_img = "\\\\?\\" + path_to_img
                filename = filename[:filename.find("%")] + ".jpg"
            print(f"Processing: {path_to_img}")
            try:
                img = load_img(path_to_img,
                               target_size=image_size,
                               interpolation="hamming")
                process_img(img, filename)
            except Exception:
                print(f"Error occurred with {path_to_img}")
                skipped += 1

print(f"--- {time.time() - start_time} seconds, skipped {skipped} files ---")
