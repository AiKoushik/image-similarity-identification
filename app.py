import os
import cv2
import shutil

def calculate_orb_similarity(imageA, imageB):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypointsA, descriptorsA = orb.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(imageB, None)

    # If no descriptors are found in either image, return 0 similarity
    if descriptorsA is None or descriptorsB is None:
        return 0

    # Match descriptors using BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptorsA, descriptorsB)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score based on the number of good matches
    similarity_score = len(matches) / max(len(keypointsA), len(keypointsB))

    return similarity_score

def rotate_image(image, angle):
    # Rotate the image by the given angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crop_center(image, crop_percent):
    # Crop the center of the image by the given percentage
    (h, w) = image.shape[:2]
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]

def compare_all_images(folder1, folder2):
    # Ensure both folders exist
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print("One or both folders do not exist.")
        return

    # Get the list of images in both folders
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))
    similarity_results = []

    # Create the destination folder if it doesn't exist
    destination_folder = os.path.join(folder1, 'similar')
    os.makedirs(destination_folder, exist_ok=True)

    # Compare each image in folder1 with each image in folder2
    for img1 in images1:
        imageA = cv2.imread(os.path.join(folder1, img1))
        if imageA is None:
            print(f"Could not read image: {img1}")
            continue

        # Crop imageA to 40% from the center
        cropped_imageA = crop_center(imageA, 1.0)
        max_similarity = 0
        best_match_img = None
        
        for img2 in images2:
            print(f"Comparing similarity between {img1} and {img2}")
            imageB = cv2.imread(os.path.join(folder2, img2))
            if imageB is None:
                print(f"Could not read image: {img2}")
                continue
            
            # Crop imageB to 40% from the center
            cropped_imageB = crop_center(imageB, 0.40)
            
            # Rotate imageB by different angles and compute similarity
            for angle in [0]:
                rotated_imageB = rotate_image(cropped_imageB, angle)
                similarity_score = calculate_orb_similarity(cropped_imageA, rotated_imageB)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match_img = img2

        similarity_percentage = max_similarity * 100  # Convert to percentage
        similarity_results.append((img1, best_match_img, similarity_percentage))

        if best_match_img:
            # Copy the image from Folder4 to the destination folder, rename it to img1's name
            source_image_path = os.path.join(folder2, best_match_img)
            destination_image_path = os.path.join(destination_folder, img1)
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied {best_match_img} as {img1} to {destination_folder}")

        print(f"Max similarity between {img1} and {best_match_img}: {max_similarity:.2f}%")

    # Sort results by similarity percentage in descending order
    similarity_results.sort(key=lambda x: x[2], reverse=True)

    # Output results
    for img1, best_match_img, similarity in similarity_results:
        print(f"Similarity between {img1} and {best_match_img}: {similarity:.2f}%")

    # Save results to a file
    output_file = 'similarity_results.txt'
    with open(output_file, 'w') as f:
        for img1, best_match_img, similarity in similarity_results:
            f.write(f"{img1}, {best_match_img}, {similarity:.2f}%\n")

    print(f"\nSimilarity results saved to {output_file}")

# Define the paths to your image folders
folder1 = 'path/to/base/images/folder'
folder2 = 'path/to/transformed/images/folder'

# Run the comparison
compare_all_images(folder1, folder2)


#python app.py