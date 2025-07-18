import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

"""----------------------------------------Split here for ipynb--------------------------------------"""


def image_to_blocks(img, block_size):
    img_height, img_width, _ = img.shape
    blocks = []
    for y in range(0, img_height, block_size[1]):
        for x in range(0, img_width, block_size[0]):
            block = img[y:y+block_size[1], x:x+block_size[0]]
            blocks.append(block)
    return blocks

def calculate_average_color(blocks):
    average_colors = []
    for block in blocks:
        average_color = np.mean(block, axis=(0, 1))
        average_colors.append(average_color)
    return np.array(average_colors)

def reconstruct_image_from_clusters(cluster_assignments, cluster_centers, block_size, frame_shape):
    reconstructed_img = np.zeros(frame_shape, dtype=np.uint8)
    block_index = 0
    for y in range(0, frame_shape[0], block_size[1]):
        for x in range(0, frame_shape[1], block_size[0]):
            if y + block_size[1] <= frame_shape[0] and x + block_size[0] <= frame_shape[1]:
                reconstructed_img[y:y + block_size[1], x:x + block_size[0]] = cluster_centers[cluster_assignments[block_index]]
            block_index += 1
    return reconstructed_img

def calculate_motion_vectors(first_frame, next_frame, block_size):
    blocks_first = image_to_blocks(first_frame, block_size)
    motion_vectors = []
    for idx, block in enumerate(blocks_first):
        block_y = (idx // (first_frame.shape[1] // block_size[0])) * block_size[1]
        block_x = (idx % (first_frame.shape[1] // block_size[0])) * block_size[0]
        mv = diamond_search(block, next_frame, block_x, block_y, 5)
        motion_vectors.append(mv)
    return motion_vectors

def calculate_residual_frame(actual_frame, predicted_frame):
    """Calculate the residual (difference) frame."""
    residual = cv2.subtract(actual_frame, predicted_frame)
    return residual

def apply_residual_frame(predicted_frame, residual):
    """Apply the residual frame to the predicted frame."""
    reconstructed_frame = cv2.add(predicted_frame, residual)
    return reconstructed_frame

def encode_video_data(cluster_assignments, motion_vectors, block_size, frame_shape):
    motion_model = np.array(motion_vectors).flatten()
    return {
        'clusters': cluster_assignments.tolist(),
        'motion_model': motion_model.tolist(),
        'block_size': block_size,
        'frame_shape': frame_shape
    }

def apply_motion_vectors_to_frame(frame, motion_vectors, block_size):
    new_frame = np.zeros_like(frame)
    num_blocks_y, num_blocks_x = frame.shape[0] // block_size[1], frame.shape[1] // block_size[0]

    for idx, motion_vector in enumerate(motion_vectors):
        block_y = (idx // num_blocks_x) * block_size[1]
        block_x = (idx % num_blocks_x) * block_size[0]
        new_block_y, new_block_x = block_y + motion_vector[1], block_x + motion_vector[0]

        if 0 <= new_block_x < frame.shape[1] - block_size[0] and 0 <= new_block_y < frame.shape[0] - block_size[1]:
            new_frame[new_block_y:new_block_y + block_size[1], new_block_x:new_block_x + block_size[0]] = frame[block_y:block_y + block_size[1], block_x:block_x + block_size[0]]

    return new_frame

def diamond_search(block, ref_frame, block_pos_x, block_pos_y, max_search_range):
    block_height, block_width, _ = block.shape
    ref_height, ref_width, _ = ref_frame.shape

    small_diamond = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]
    large_diamond = [(-2, 0), (0, -2), (2, 0), (0, 2)] + small_diamond

    def get_sad(center_x, center_y):
        y1, y2 = center_y, center_y + block_height
        x1, x2 = center_x, center_x + block_width
        if 0 <= x1 < ref_width and 0 <= y1 < ref_height and x2 <= ref_width and y2 <= ref_height:
            candidate_block = ref_frame[y1:y2, x1:x2]
            return np.sum(np.abs(block.astype(np.int32) - candidate_block.astype(np.int32)))
        else:
            return float('inf')

    min_sad = float('inf')
    motion_vector = (0, 0)

    center_x, center_y = block_pos_x, block_pos_y

    # Large diamond search
    for dx, dy in large_diamond:
        x, y = center_x + dx, center_y + dy
        sad = get_sad(x, y)
        if sad < min_sad:
            min_sad = sad
            motion_vector = (x - block_pos_x, y - block_pos_y)

    # Refine with small diamond search
    center_x, center_y = block_pos_x + motion_vector[0], block_pos_y + motion_vector[1]
    for dx, dy in small_diamond:
        x, y = center_x + dx, center_y + dy
        sad = get_sad(x, y)
        if sad < min_sad:
            min_sad = sad
            motion_vector = (x - block_pos_x, y - block_pos_y)

    return motion_vector


"""----------------------------------------Split here for ipynb--------------------------------------"""


#Add location of video in video_path
video_path = "/content/drive/MyDrive/VIDEO_COMPRESSION/VIDEOS/AlitaBattleAngel.mkv"
block_size = (16, 16)
num_clusters = 5
seconds_to_process = 60
frame_rate = 24
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")


ret, first_frame = cap.read()
if not ret:
    raise IOError("Cannot read the first frame")

blocks_first = image_to_blocks(first_frame, block_size)
average_colors_first = calculate_average_color(blocks_first)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(average_colors_first)
cluster_assignments_first = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

original_frames = [first_frame]
compressed_data = []
count = 0


"""----------------------------------------Split here for ipynb--------------------------------------"""


# First pass to process video and store compressed data
for _ in range(seconds_to_process * frame_rate - 1):
    ret, next_frame = cap.read()
    if not ret:
        break

    count += 1
    print(f"Processing frame {count}")
    original_frames.append(next_frame)
    motion_vectors = calculate_motion_vectors(first_frame, next_frame, block_size)
    encoded_data = encode_video_data(cluster_assignments_first, motion_vectors, block_size, first_frame.shape)
    compressed_data.append(encoded_data)

    first_frame = next_frame

total_frames = len(original_frames) - 1
compressed_data = []

# Second pass to calculate residuals and compress
for i in range(1, len(original_frames)):
    current_frame = original_frames[i - 1]
    next_frame = original_frames[i]
    motion_vectors = calculate_motion_vectors(current_frame, next_frame, block_size)
    predicted_frame = apply_motion_vectors_to_frame(current_frame, motion_vectors, block_size)
    residual_frame = calculate_residual_frame(next_frame, predicted_frame)

    # Use numpy array instead of list for residual
    encoded_data = encode_video_data(cluster_assignments_first, motion_vectors, block_size, current_frame.shape)
    encoded_data['residual'] = residual_frame  # Store as numpy array
    compressed_data.append(encoded_data)

    progress_percentage = (i / total_frames) * 100
    print(f"Compressing frame {i}/{total_frames} ({progress_percentage:.2f}%)")

# Save compressed_data to a file
compressed_data_path = 'compressed_data.pkl'

with open(compressed_data_path, 'wb') as file:
    pickle.dump(compressed_data, file)

print(f"Compressed data saved to {compressed_data_path}")

compressed_data_path = 'compressed_data.pkl'


"""----------------------------------------Split here for ipynb--------------------------------------"""


# Load compressed_data from the file
with open(compressed_data_path, 'rb') as file:
    compressed_data = pickle.load(file)

print("Compressed data successfully loaded.")

refresh_rate = 4
def refresh_needed(frame_count, refresh_rate):
    return frame_count % refresh_rate == 0
decompressed_frames = []
reference_frame = original_frames[0]
frame_count = 0

for encoded_data in compressed_data:
    frame_count += 1
    motion_vectors = np.array(encoded_data['motion_model']).reshape(-1, 2)
    residual_frame = np.array(encoded_data['residual'], dtype=np.uint8)

    predicted_frame = apply_motion_vectors_to_frame(reference_frame, motion_vectors, block_size)
    reconstructed_frame = apply_residual_frame(predicted_frame, residual_frame)

    decompressed_frames.append(reconstructed_frame)

    if refresh_needed(frame_count, refresh_rate):
        reference_frame = original_frames[min(frame_count, len(original_frames) - 1)]
    else:
        reference_frame = reconstructed_frame

import cv2

output_video_path = 'reconstructed_video2.avi'
codec = cv2.VideoWriter_fourcc(*'XVID')
output_frame_rate = frame_rate

print(f"Total frames to write: {len(decompressed_frames)}")
if decompressed_frames:
    print(f"Frame size: {decompressed_frames[0].shape}")

output_size = (decompressed_frames[0].shape[1], decompressed_frames[0].shape[0])
out = cv2.VideoWriter(output_video_path, codec, output_frame_rate, output_size)

for idx, frame in enumerate(decompressed_frames):
    print(f"Writing frame {idx + 1}/{len(decompressed_frames)}")
    out.write(frame)

out.release()

print(f"Video reconstructed and saved to {output_video_path}")


"""----------------------------------------Split here for ipynb--------------------------------------"""


snr_values = [compare_psnr(orig, decomp) for orig, decomp in zip(original_frames, decompressed_frames)]
average_snr = sum(snr_values) / len(snr_values)
cap.release()

print(f"Average SNR: {average_snr} dB")


"""----------------------------------------------- END ----------------------------------------------"""