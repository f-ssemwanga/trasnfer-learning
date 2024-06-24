from random import shuffle, choice, randint
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2
import math


def load_rgb_data(
    IMAGE_DIRECTORY,
    IMAGE_SIZE,
    directory_depth=2,
    max_number_of_images=None,
    shuffle_data=True,
):
    """
    Load RGB images from the specified directory and subdirectories.

    Args:
    - IMAGE_DIRECTORY (str): The root directory containing images.
    - IMAGE_SIZE (int): The target size to resize images (square).
    - directory_depth (int): The depth of directory to use as label.
    - max_number_of_images (int): Maximum number of images to load.
    - shuffle_data (bool): Whether to shuffle the data.

    Returns:
    - images (numpy.ndarray): Loaded and processed images.
    - labels (numpy.ndarray): Corresponding labels.
    """
    print("Loading images...")
    data = []
    count = 0

    for folder, _, file_names in os.walk(IMAGE_DIRECTORY):
        if len(file_names) != 0:
            label = (
                folder.split(os.sep)[-directory_depth]
                if directory_depth > 0
                else os.path.basename(folder)
            )
            for image_name in file_names:
                image_path = os.path.join(folder, image_name)
                if ".DS_Store" not in image_path:
                    try:
                        img = Image.open(image_path).convert("RGB")
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                        data.append([np.array(img), label])
                        count += 1
                        if count % 50 == 0:
                            print(f"Loaded {count} images so far ...")
                        if max_number_of_images and count >= max_number_of_images:
                            print("Reached maximum number of images. Preparing data...")
                            return _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY)
                    except Exception as e:
                        print(f"Cannot load {image_path}: {e}")

    print(f"Number of images loaded: {count}")
    if shuffle_data:
        random.shuffle(data)
        print("Dataset shuffled.")

    print("Preparing data after full iteration...")
    return _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY)


def _prepare_data(data, IMAGE_SIZE, IMAGE_DIRECTORY):
    """
    Prepare images and labels as numpy arrays.

    Args:
    - data (list): List of image and label pairs.
    - IMAGE_SIZE (int): Size of the image.

    Returns:
    - images (numpy.ndarray): Processed images.
    - labels (numpy.ndarray): Corresponding labels.
    """
    print("Convert images and labels into a numpy array")
    images = np.array([item[0] for item in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    labels = np.array([item[1] for item in data])
    print(f"Data Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Unique Labels: {np.unique(labels)}")
    print(f"Loading dataset completed from path: {IMAGE_DIRECTORY}")
    return images, labels


def normalize_data(dataset):
    """
    Normalize dataset to range [0, 1].

    Args:
    - dataset (numpy.ndarray): Dataset to normalize.

    Returns:
    - dataset (numpy.ndarray): Normalized dataset.
    """
    print("Normalizing data...")
    return dataset / 255.0


def display_image(trainX, trainY, index=0):
    """
    Display a single image with its label.

    Args:
    - trainX (numpy.ndarray): Dataset of images.
    - trainY (numpy.ndarray): Corresponding labels.
    - index (int): Index of the image to display.
    """
    plt.imshow(trainX[index])
    print("Label =", np.squeeze(trainY[index]))
    print("Image shape:", trainX[index].shape)


def display_one_image(one_image, its_label):
    """
    Display a single image with its label.

    Args:
    - one_image (numpy.ndarray): Image to display.
    - its_label (str): Label of the image.
    """
    plt.imshow(one_image)
    print("Label =", its_label)
    print("Image shape:", one_image.shape)


def display_dataset_shape(X, Y):
    """
    Display the shapes of the dataset and labels.

    Args:
    - X (numpy.ndarray): Dataset of images.
    - Y (numpy.ndarray): Corresponding labels.
    """
    print("Shape of images:", X.shape)
    print("Shape of labels:", Y.shape)


def plot_sample_from_dataset(images, labels, rows=5, columns=5, width=8, height=8):
    """
    Plot a sample grid of images from the dataset.

    Args:
    - images (numpy.ndarray): Dataset of images.
    - labels (numpy.ndarray): Corresponding labels.
    - rows (int): Number of rows in the grid.
    - columns (int): Number of columns in the grid.
    - width (int): Width of the figure.
    - height (int): Height of the figure.
    """
    plt.figure(figsize=(width, height))
    for i in range(rows * columns):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.show()


def display_dataset_folders(path):
    """
    Display sorted list of folders in the dataset directory.

    Args:
    - path (str): Path to the dataset directory.
    """
    classes = sorted(os.listdir(path))
    print(classes)


def get_data_distribution(
    IMAGE_DIRECTORY, directory_depth=2, output_file=None, plot_stats=True
):
    print("Loading images...")
    # list structure to collect the statistics
    stats = []

    for folder, directory_name, file_name in os.walk(IMAGE_DIRECTORY):
        if len(file_name) != 0:
            if directory_depth != 0:
                # print (folder, directory_name, file_name)
                folders_list = folder.split("/")[0:-1]
            else:
                folders_list = folder.split("/")[-1:]

            # print(folders_list)
            label = folders_list[-directory_depth]
            # print (label)
            for i in range(len(file_name)):
                image_name = file_name[i]
                image_path = os.path.join(folder, image_name)
                if ".DS_Store" not in image_path:
                    # print(image_path)
                    try:
                        img = Image.open(image_path)
                        # print(image_path)
                        rgbimg = Image.new("RGB", img.size)
                        rgbimg.paste(img)
                        img = rgbimg

                        width, height = img.size
                        # get the size of the image in KB
                        size_kb = os.stat(image_path).st_size / 1000
                        # add the size to a list of sizes to be
                        stats.append(
                            [
                                label,
                                os.path.basename(image_path),
                                width,
                                height,
                                size_kb,
                            ]
                        )
                        # print(stats)
                    except Exception:
                        pass

    if output_file is not None:
        # convert the list into a dataframe to make it easy to save into a CSV
        stats_dataframe = pd.DataFrame(
            stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
        )
        stats_dataframe.to_csv(output_file, index=False)
        print("Stats collected and saved in .", output_file)
    else:
        print("Stats collected")

    return stats


"""
def plot_dataset_distribution(
    stats,
    num_cols=5,
    width=10,
    height=5,
    histogram_bins=10,
    histogram_range=[0, 1000],
    figure_padding=4,
):
    import matplotlib.pyplot as plt
    import math

    stats_df = pd.DataFrame(
        stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
    )
    list_sizes = stats_df["Size_in_KB"]
    number_of_classes = stats_df["Class"].nunique()
    print(f"{number_of_classes} classes found in the dataset")

    class_names = ["Whole dataset"]
    list_sizes_per_class = [list_sizes]

    for class_name in stats_df["Class"].unique():
        class_avg_size = list_sizes[stats_df["Class"] == class_name].mean()
        print(f"Sizes of class {class_name} have an average size of {class_avg_size}")
        list_sizes_per_class.append(list_sizes[stats_df["Class"] == class_name])
        class_names.append(class_name)

    class_count_dict = stats_df["Class"].value_counts().to_dict()
    print(class_count_dict)

    num_rows = math.ceil((number_of_classes + 1) / num_cols)
    if number_of_classes < num_cols:
        num_cols = number_of_classes + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    fig.tight_layout(pad=figure_padding)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = num_cols * i + j
            if idx < len(list_sizes_per_class):
                axes[i, j].hist(
                    list_sizes_per_class[idx],
                    bins=histogram_bins,
                    range=histogram_range,
                )
                axes[i, j].set_xlabel("Image size (in KB)", fontweight="bold")
                axes[i, j].set_title(f"{class_names[idx]} images", fontweight="bold")

    plt.figure()
    plt.bar(class_count_dict.keys(), class_count_dict.values())
    for index, class_name in enumerate(class_count_dict):
        plt.text(
            index,
            class_count_dict[class_name] + 1,
            str(class_count_dict[class_name]),
            ha="center",
        )
    plt.show()"""


def plot_dataset_distribution(
    stats,
    num_cols=5,
    width=10,
    height=5,
    histogram_bins=10,
    histogram_range=[0, 1000],
    figure_padding=4,
):
    # Convert the list into a dataframe
    stats_frame = pd.DataFrame(
        stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
    )

    # Extract the dataframe related to sizes only
    list_sizes = stats_frame["Size_in_KB"]

    # Get the number of classes in the dataset
    number_of_classes = stats_frame["Class"].nunique()
    print(number_of_classes, " classes found in the dataset")

    # Create a list of (list of sizes) for each class of images
    # Start with the sizes of all images in the dataset
    list_sizes_per_class = [list_sizes]
    class_names = ["whole dataset"]
    print("Images of the whole dataset have an average size of ", list_sizes.mean())

    for c in stats_frame["Class"].unique():
        print(
            "sizes of class [",
            c,
            "] have an average size of ",
            list_sizes.loc[stats_frame["Class"] == c].mean(),
        )
        # Append the sizes of images of a particular class
        list_sizes_per_class.append(list_sizes.loc[stats_frame["Class"] == c])
        class_names.append(c)

    class_count_dict = {}
    for c in stats_frame["Class"].unique():
        print(
            "number of instances in class [",
            c,
            "] is ",
            stats_frame.loc[stats_frame["Class"] == c].count()["Class"],
        )
        class_count_dict[c] = stats_frame.loc[stats_frame["Class"] == c].count()[
            "Class"
        ]

    num_rows = math.ceil((number_of_classes + 1) / num_cols)
    if number_of_classes < num_cols:
        num_cols = number_of_classes + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    fig.tight_layout(pad=figure_padding)

    class_count = 0
    if num_rows == 1 or num_cols == 1:
        for i in range(num_rows):
            for j in range(num_cols):
                axes[j + i].hist(
                    list_sizes_per_class[num_cols * i + j],
                    bins=histogram_bins,
                    range=histogram_range,
                )
                axes[j + i].set_xlabel("Image size (in KB)", fontweight="bold")
                axes[i + j].set_title(
                    class_names[j + i] + " images ", fontweight="bold"
                )
                class_count += 1
                if class_count == number_of_classes + 1:
                    break

    else:
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i, j].hist(
                    list_sizes_per_class[num_cols * i + j],
                    bins=histogram_bins,
                    range=histogram_range,
                )
                axes[i, j].set_xlabel("Image size (in KB)", fontweight="bold")
                axes[i, j].set_title(
                    class_names[num_cols * i + j] + " images ", fontweight="bold"
                )
                class_count += 1
                if class_count == number_of_classes + 1:
                    break

    plt.figure()  # Create a new figure for the bar chart
    print(class_count_dict)
    plt.bar(*zip(*class_count_dict.items()))

    for index, car_brand in enumerate(class_count_dict):
        plt.text(
            index,
            class_count_dict[car_brand] + 1,
            str(class_count_dict[car_brand]),
            ha="center",
        )
    plt.show()
