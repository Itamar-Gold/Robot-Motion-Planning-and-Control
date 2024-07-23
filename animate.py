import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots(figsize=(14, 7))

# Create a list to store the frames
ims = []

# For real time capturing
t_span = (0, 10)  # 10 seconds simulation
t = np.linspace(t_span[0], t_span[1], 1000, endpoint=False)
# Define the base path to the folder containing the images
base_folder_path = 'images'  # first folder
solution_title = 'time_seq'  # subfolder containing the images

# Loop through the specified range to read images and add to the list of frames
for i in t:

    if i <= 6:
        path = os.path.join(base_folder_path, solution_title, f'Time_{i}.png')
        if os.path.exists(path):
            img = plt.imread(path)
            im = ax.imshow(img, animated=True)
            ims.append([im])

# Create an animation
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

# Define the output path for the GIF
output_path = os.path.join(base_folder_path, 'realtime_simulation2.gif')

# Save the animation as a GIF
ani.save(output_path, writer='Pillow')

print(f'GIF saved at {output_path}')
