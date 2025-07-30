#%% Open and visualise '../data/celeba/img_align_celeba/202600.jpg'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
def display_image(image_path):
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        # plt.show()
    # else:
    #     print(f"Image not found: {image_path}")

    ## Save as jpg here
    new_image_path = "./202600.jpg"
    mpimg.imsave(new_image_path, img)
    print(f"Image saved as: {new_image_path}")

# Example usage
image_path = '../data/celeba/img_align_celeba/202600.jpg'
display_image(image_path)
