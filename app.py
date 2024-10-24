import os
import cv2
import shutil
import argparse
import asyncio
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# Define similarity calculation methods
def calculate_ssim(imageA, imageB):
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        imageA (numpy.ndarray): The first input image.
        imageB (numpy.ndarray): The second input image.

    Returns:
        float: The SSIM value between the two images.
    """
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    return ssim(grayA, grayB)

def calculate_orb(imageA, imageB):
    """
    Calculate similarity between two images using ORB.

    Args:
        imageA (numpy.ndarray): The first input image.
        imageB (numpy.ndarray): The second input image.

    Returns:
        float: The similarity score based on ORB features. Returns 0 if no descriptors are found.
    """
    orb = cv2.ORB_create()
    keypointsA, descriptorsA = orb.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(imageB, None)

    # If no descriptors are found in either image, return 0 similarity
    if descriptorsA is None or descriptorsB is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptorsA, descriptorsB)
    matches = sorted(matches, key=lambda x: x.distance)

    similarity_score = len(matches) / max(len(keypointsA), len(keypointsB)) if keypointsA and keypointsB else 0

    return similarity_score


def calculate_mse(imageA, imageB):
    """
    Calculate Mean Squared Error (MSE) between two images.

    Args:
        imageA (numpy.ndarray): The first input image.
        imageB (numpy.ndarray): The second input image.

    Returns:
        float: The Mean Squared Error between the two images.
    """
    return ((imageA - imageB) ** 2).mean()

def crop_from_center(image, percentage):
    """
    Crop the input image from the center by a specified percentage.

    Args:
        image (numpy.ndarray): The input image to crop. It should be in the format of a NumPy array.
        percentage (float): The percentage of the original image dimensions to retain after cropping. 
                           Value should be between 0 and 1, where 1 means no cropping and 0 means no image.

    Returns:
        numpy.ndarray: The cropped image as a NumPy array. If the percentage is less than or equal to 0,
                       returns an empty array. If the percentage is greater than or equal to 1, returns
                       the original image.

    Raises:
        ValueError: If the input percentage is not in the range (0, 1].
    """
    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be between 0 (exclusive) and 1 (inclusive).")

    # Get the original dimensions of the image
    height, width = image.shape[:2]

    # Calculate the dimensions of the new cropped image
    new_height = int(height * percentage)
    new_width = int(width * percentage)
    
    # Calculate the starting coordinates for cropping
    start_x = width // 2 - new_width // 2  # Center crop along width
    start_y = height // 2 - new_height // 2  # Center crop along height
    
    # Crop the image and return the cropped section
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

    return cropped_image


def rotate_image(imageB, angle):
    """
    Rotate the image by the specified angle.

    Args:
        imageB (numpy.ndarray): The input image to rotate.
        angle (int): The angle to rotate the image (0, 90, 180, 270 degrees).

    Returns:
        numpy.ndarray: The rotated image.
    """
    if angle == 0:
        return imageB
    elif angle == 90:
        return cv2.rotate(imageB, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(imageB, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(imageB, cv2.ROTATE_90_COUNTERCLOCKWISE)

async def copy_best_match_image(best_match_img, img1, destination_folder, target_matching_folder):
    """
    Copy the best match image from the target folder to the destination folder, renamed as img1.

    Args:
        best_match_img (str): The filename of the best matched image.
        img1 (str): The filename of the original image.
        destination_folder (str): The destination folder path to copy the image to.
        target_matching_folder (str): The path of the target matching folder.
    """
    if best_match_img:
        source_image_path = os.path.join(target_matching_folder, best_match_img)
        destination_image_path = os.path.join(destination_folder, img1)
        shutil.copy(source_image_path, destination_image_path)
        print(f"Copied {best_match_img} as {img1} to {destination_folder}")

async def compare_all_images(original_folder, target_matching_folder, is_rotation, similarity_function, crop_percentage, save_result, copy_images):
    """
    Compare images between two folders using the chosen similarity method.

    Args:
        original_folder (str): Path to the original folder containing images.
        target_matching_folder (str): Path to the target matching folder containing images.
        is_rotation (bool): Flag to indicate whether to apply rotation comparison.
        similarity_function (callable): The function used to calculate similarity.
        crop_percentage (float): Percentage to crop images from the center.
        save_result (bool): Flag to save comparison results to a CSV file.
    """
    if not os.path.exists(original_folder) or not os.path.exists(target_matching_folder):
        print("One or both folders do not exist.")
        return

    images1 = sorted(os.listdir(original_folder))
    images2 = sorted(os.listdir(target_matching_folder))
    result_file = None
    if save_result:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        result_file = f'result_{timestamp}.csv'
        with open(result_file, 'w') as f:
            f.write("Original Image,Matched Image,Similarity Percentage,Rotation Angle\n")

    destination_folder = os.path.join(target_matching_folder, 'target_matched')
    os.makedirs(destination_folder, exist_ok=True)

    for img1 in images1:
        imageA = cv2.imread(os.path.join(original_folder, img1))
        if imageA is None:
            print(f"Could not read image: {img1}")
            continue
        
        # Only call crop_from_center if crop_percentage is provided
        if crop_percentage is not None:
            imageA = crop_from_center(imageA, crop_percentage)

        max_similarity = 0
        best_match_img = None
        best_rotation = 0

        for img2 in images2:
            imageB = cv2.imread(os.path.join(target_matching_folder, img2))
            if imageB is None:
                continue
            
            # Only call crop_from_center if crop_percentage is provided
            if crop_percentage is not None:
                imageB = crop_from_center(imageB, crop_percentage)

            for angle in ([0, 90, 180, 270] if is_rotation else [0]):
                rotated_imageB = rotate_image(imageB, angle)
                similarity_score = similarity_function(imageA, rotated_imageB)
                
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match_img = img2
                    best_rotation = angle

            print(f"Max similarity between {img1} and {img2}: {max_similarity:.2f}")

        if copy_images and best_match_img:
            await copy_best_match_image(best_match_img, img1, destination_folder, target_matching_folder)
            images2.remove(best_match_img)

        if save_result and best_match_img:
            with open(result_file, 'a') as f:
                f.write(f"{img1},{best_match_img},{max_similarity:.2f},{best_rotation}\n")
            print(f"Saved result for {img1} to {result_file}")

def similarity_lambda(method="orb"):
    """
    Return a lambda function for the chosen similarity method.

    Args:
        method (str): The similarity method to use (either "orb", "ssim", or "mse").

    Returns:
        callable: A function that calculates similarity between two images.
    """
    if method == "orb":
        return lambda imageA, imageB: calculate_orb(imageA, imageB)
    elif method == "mse":
        return lambda imageA, imageB: calculate_mse(imageA, imageB)
    else:  # default to SSIM
        return lambda imageA, imageB: calculate_ssim(imageA, imageB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare images between two folders.")
    parser.add_argument("--original_folder", required=True, help="Path to the original folder containing images")
    parser.add_argument("--target_matching_folder", required=True, help="Path to the target matching folder containing images")
    parser.add_argument("--is_rotation", action='store_true', help="Apply rotation comparison")
    parser.add_argument("--method", choices=["orb", "ssim", "mse"], default="orb", help="Similarity calculation method to use")
    parser.add_argument("--crop_percentage", type=float, default=None, help="Crop images from the center by a percentage (None for no crop)")
    parser.add_argument("--save_result", type=bool, nargs='?', const=True, default=True, help="Flag to save the comparison results to a CSV file")
    parser.add_argument('--copy_images', action='store_true', help="Enable copying of best match images")

    args = parser.parse_args()

    similarity_function = similarity_lambda(args.method)

    asyncio.run(compare_all_images(args.original_folder, args.target_matching_folder, args.is_rotation, similarity_function, args.crop_percentage, args.save_result, args.copy_images))


#python app.py --original_folder <path_to_original_folder> --target_matching_folder <path_to_target_matching_folder> [--is_rotation] [--method <orb|ssim|mse>] [--crop_percentage <percentage>] [--save_result] [--copy_images]

