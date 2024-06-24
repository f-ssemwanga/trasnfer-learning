from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2
import math
from random import randint


def load_rgb_data(IMAGE_DIRECTORY, IMAGE_SIZE, shuffle=True):
    print("Loading images...")
    data = []
    directories = next(os.walk(IMAGE_DIRECTORY))[1]
    print(directories)
    for directory_name in directories:
        print(f"Loading {directory_name}")
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, directory_name)))[2]
        print(f"Loading {len(file_names)} files from {directory_name} class ...")
        for image_name in file_names:
            if ".DS_Store" not in image_name:
                image_path = os.path.join(IMAGE_DIRECTORY, directory_name, image_name)
                label = directory_name
                img = Image.open(image_path).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                data.append([np.array(img), label])

    if shuffle:
        random.shuffle(data)
    training_images = np.array([i[0] for i in data]).reshape(
        -1, IMAGE_SIZE, IMAGE_SIZE, 3
    )
    training_labels = np.array([i[1] for i in data])

    print("File loading completed.")
    return training_images, training_labels


def load_rgb_data_cv(IMAGE_DIRECTORY, IMAGE_SIZE, shuffle=True):
    print("Loading images...")
    data = []
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for directory_name in directories:
        print(f"Loading {directory_name}")
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, directory_name)))[2]
        print(f"Loading {len(file_names)} files from {directory_name} class ...")
        for image_name in file_names:
            image_path = os.path.join(IMAGE_DIRECTORY, directory_name, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            label = directory_name
            data.append([np.array(img), label])

    if shuffle:
        random.shuffle(data)
    training_images = np.array([i[0] for i in data]).reshape(
        -1, IMAGE_SIZE, IMAGE_SIZE, 3
    )
    training_labels = np.array([i[1] for i in data])

    print("File loading completed.")
    return training_images, training_labels


def normalize_data(dataset):
    print("Normalize data")
    return dataset / 255.0


def display_image(trainX, trainY, index=0):
    plt.imshow(trainX[index])
    print(f"Label = {str(np.squeeze(trainY[index]))}")
    print(f"image shape: {trainX[index].shape}")


def display_one_image(one_image, its_label):
    plt.imshow(one_image)
    print(f"Label = {its_label}")
    print(f"image shape: {one_image.shape}")


def display_dataset_shape(X, Y):
    print("Shape of images:", X.shape)
    print("Shape of labels:", Y.shape)


def plot_sample_from_dataset(images, labels, rows=5, columns=5, width=8, height=8):
    plt.figure(figsize=(width, height))
    for i in range(rows * columns):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()


def display_dataset_folders(path):
    classes = os.listdir(path)
    classes.sort()
    print(classes)


def get_data_distribution(IMAGE_DIRECTORY, output_file=None, plot_stats=True):
    print("Loading images...")
    stats = []
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for directory_name in directories:
        print(f"Loading {directory_name}")
        images_file_names = next(
            os.walk(os.path.join(IMAGE_DIRECTORY, directory_name))
        )[2]
        print(f"Loading {len(images_file_names)} files from {directory_name} class ...")
        for image_name in images_file_names:
            image_path = os.path.join(IMAGE_DIRECTORY, directory_name, image_name)
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            size_kb = os.stat(image_path).st_size / 1000
            stats.append(
                [directory_name, os.path.basename(image_name), width, height, size_kb]
            )

    if output_file is not None:
        stats_dataframe = pd.DataFrame(
            stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
        )
        stats_dataframe.to_csv(output_file, index=False)
        print(f"Stats collected and saved in {output_file}")
    else:
        print("Stats collected")

    return stats


def plot_dataset_distribution(
    stats,
    num_cols=5,
    width=10,
    height=5,
    histogram_bins=10,
    histogram_range=[0, 1000],
    figure_padding=4,
):
    stats_frame = pd.DataFrame(
        stats, columns=["Class", "Filename", "Width", "Height", "Size_in_KB"]
    )
    list_sizes = stats_frame["Size_in_KB"]
    number_of_classes = stats_frame["Class"].nunique()
    print(f"{number_of_classes} classes found in the dataset")

    list_sizes_per_class = [list_sizes]
    class_names = ["whole dataset"]
    print(f"Images of the whole dataset have an average size of {list_sizes.mean()}")

    for c in stats_frame["Class"].unique():
        print(
            f"Sizes of class [{c}] have an average size of {list_sizes.loc[stats_frame['Class'] == c].mean()}"
        )
        list_sizes_per_class.append(list_sizes.loc[stats_frame["Class"] == c])
        class_names.append(c)

    class_count_dict = {
        c: stats_frame.loc[stats_frame["Class"] == c].count()["Class"]
        for c in stats_frame["Class"].unique()
    }
    for c, count in class_count_dict.items():
        print(f"Number of instances in class [{c}] is {count}")

    num_rows = math.ceil((number_of_classes + 1) / num_cols)
    if number_of_classes < num_cols:
        num_cols = number_of_classes + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    fig.tight_layout(pad=figure_padding)
    class_count = 0

    for i in range(num_rows):
        for j in range(num_cols):
            if class_count == number_of_classes + 1:
                break
            axes[i, j].hist(
                list_sizes_per_class[class_count],
                bins=histogram_bins,
                range=histogram_range,
            )
            axes[i, j].set_xlabel("Image size (in KB)", fontweight="bold")
            axes[i, j].set_title(
                f"{class_names[class_count]} images", fontweight="bold"
            )
            class_count += 1

    plt.figure()
    plt.bar(class_count_dict.keys(), class_count_dict.values())
    for index, count in enumerate(class_count_dict.values()):
        plt.text(index, count + 1, str(count), ha="center")
    plt.show()


def reshape_image_for_neural_network_input(image, IMAGE_SIZE=224, normalize=True):
    print("Flatten the image")
    image = np.reshape(image, [IMAGE_SIZE * IMAGE_SIZE * 3, 1])
    print(f"image.shape {image.shape}")
    print("Reshape the image to be similar to the input feature vector")
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float")
    print(f"image.shape {image.shape}")
    if normalize:
        image = image / 255.0
    return image


def plot_loss_accuracy(H, EPOCHS, output_file=None):
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


def plot_loss_accuracy_from_csv(log_history_filename, output_file=None):
    history_dataframe = pd.read_csv(log_history_filename)
    N = history_dataframe.shape[0]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history_dataframe["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history_dataframe["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history_dataframe["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history_dataframe["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


def draw_accuracy_graph(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def draw_loss_graph(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def get_cars_classes_dict(testY, testLabels):
    number_of_classes = np.unique(testLabels).size
    car_dict = {np.argmax(label): testLabels[i] for i, label in enumerate(testY)}
    return car_dict, number_of_classes


def plot_test_image(
    testX, image_index, predictions_array, true_binary_labels, car_dict=None
):
    single_predictions_array, true_binary_label, test_image = (
        predictions_array,
        true_binary_labels[image_index],
        testX[image_index],
    )
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_image, cmap=plt.cm.binary)

    predicted_binary_label = np.argmax(predictions_array)
    color = "blue" if predicted_binary_label == true_binary_label else "red"

    if car_dict is None:
        plt.xlabel(
            f"{image_index} predicted: {predicted_binary_label} {100 * np.max(single_predictions_array):2.0f}% (true: {true_binary_label})",
            color=color,
        )
    else:
        plt.xlabel(
            f"{image_index} predicted: {car_dict[predicted_binary_label]} {100 * np.max(single_predictions_array):2.0f}% (true: {car_dict[true_binary_label]})",
            color=color,
        )


def plot_value_array(i, predictions_array, true_label, number_of_classes=3):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(number_of_classes))
    plt.yticks([])
    thisplot = plt.bar(range(number_of_classes), predictions_array, color="#FFFFFF")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


def plot_sample_predictions(
    testX,
    predictions_array,
    true_binary_labels,
    number_of_classes=3,
    num_rows=10,
    num_cols=4,
    width=None,
    height=None,
    is_random=True,
):
    num_images = num_rows * num_cols
    if num_images > testX.shape[0]:
        raise Exception(
            f"num_rows*num_cols is {num_images}, must be smaller than number of images in the Test Dataset {testX.shape[0]}"
        )

    width = 6 * num_cols if width is None else width
    height = 2 * num_rows if height is None else height

    plt.figure(figsize=(width, height))
    plt.style.use(["seaborn-bright"])

    image_index = -1
    for i in range(num_images):
        image_index = randint(0, testX.shape[0] - 1) if is_random else image_index + 1
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_test_image(
            testX, image_index, predictions_array[image_index], true_binary_labels
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(
            image_index,
            predictions_array[image_index],
            true_binary_labels,
            number_of_classes,
        )
    plt.tight_layout()
    plt.show()


def plot_car_sample_predictions(
    testX,
    predictions_array,
    testY,
    number_of_classes=3,
    num_rows=10,
    num_cols=4,
    width=None,
    height=None,
    is_random=True,
):
    num_images = num_rows * num_cols
    true_binary_labels = np.argmax(testY, axis=1)
    if num_images > testX.shape[0]:
        raise Exception(
            f"num_rows*num_cols is {num_images}, must be smaller than number of images in the Test Dataset {testX.shape[0]}"
        )

    width = 6 * num_cols if width is None else width
    height = 2 * num_rows if height is None else height

    plt.figure(figsize=(width, height))
    plt.style.use(["seaborn-bright"])

    image_index = -1
    for i in range(num_images):
        image_index = randint(0, testX.shape[0] - 1) if is_random else image_index + 1
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_test_image(
            testX, image_index, predictions_array[image_index], true_binary_labels
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(
            image_index,
            predictions_array[image_index],
            true_binary_labels,
            number_of_classes,
        )
    plt.tight_layout()
    plt.show()


def plot_car_sample_predictions_v2(
    testX,
    testY,
    testLabels,
    predictions_array,
    num_rows=10,
    num_cols=4,
    width=None,
    height=None,
    is_random=True,
):
    car_dict, number_of_classes = get_cars_classes_dict(testY, testLabels)

    num_images = num_rows * num_cols
    true_binary_labels = np.argmax(testY, axis=1)
    if num_images > testX.shape[0]:
        raise Exception(
            f"num_rows*num_cols is {num_images}, must be smaller than number of images in the Test Dataset {testX.shape[0]}"
        )

    width = 6 * num_cols if width is None else width
    height = 2 * num_rows if height is None else height

    plt.figure(figsize=(width, height))
    plt.style.use(["seaborn-bright"])

    image_index = -1
    for i in range(num_images):
        image_index = randint(0, testX.shape[0] - 1) if is_random else image_index + 1
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_test_image(
            testX,
            image_index,
            predictions_array[image_index],
            true_binary_labels,
            car_dict,
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(
            image_index,
            predictions_array[image_index],
            true_binary_labels,
            number_of_classes,
        )
    plt.tight_layout()
    plt.show()


def plot_misclassifications(
    testX,
    testY,
    testLabels,
    predictions_array,
    num_rows=10,
    num_cols=4,
    width=None,
    height=None,
    is_random=True,
):
    car_dict, number_of_classes = get_cars_classes_dict(testY, testLabels)

    num_images = num_rows * num_cols
    true_binary_labels = np.argmax(testY, axis=1)
    if num_images > testX.shape[0]:
        raise Exception(
            f"num_rows*num_cols is {num_images}, must be smaller than number of images in the Test Dataset {testX.shape[0]}"
        )

    width = 6 * num_cols if width is None else width
    height = 2 * num_rows if height is None else height

    plt.figure(figsize=(width, height))
    plt.style.use(["seaborn-bright"])

    image_index = -1
    count = 0
    misclassified_indices = {}
    for i in range(num_images):
        image_index = randint(0, testX.shape[0] - 1) if is_random else image_index + 1
        predicted_binary_label = np.argmax(predictions_array[image_index])
        if predicted_binary_label != true_binary_labels[image_index]:
            plt.subplot(num_rows, 2 * num_cols, 2 * count + 1)
            plot_test_image(
                testX,
                image_index,
                predictions_array[image_index],
                true_binary_labels,
                car_dict,
            )
            plt.subplot(num_rows, 2 * num_cols, 2 * count + 2)
            plot_value_array(
                image_index,
                predictions_array[image_index],
                true_binary_labels,
                number_of_classes,
            )
            count += 1
            misclassified_indices[image_index] = [
                car_dict[true_binary_labels[image_index]],
                car_dict[predicted_binary_label],
            ]

    print(f"Number of misclassifications: {count}")
    print(f"Number of images: {num_images}")
    print(f"Rate of misclassifications: {100 * count / num_images}%")
    print(misclassified_indices)
    plt.tight_layout()
    plt.show()
    return misclassified_indices


def misclassification_stats(misclassification_dict):
    gen = 0
    model = 0
    brand = 0
    dict_size = len(misclassification_dict)
    for key in misclassification_dict:
        true_label = misclassification_dict[key][0].split("-")
        predicted_label = misclassification_dict[key][1].split("-")

        brand_of_true_label = true_label[0]
        brand_of_predicted_label = predicted_label[0]

        model_of_true_label = true_label[1]
        model_of_predicted_label = predicted_label[1]

        gen_of_true_label = (
            " ".join(true_label[3:5]) if len(true_label) == 5 else true_label[3]
        )
        gen_of_predicted_label = (
            " ".join(predicted_label[3:5])
            if len(predicted_label) == 5
            else predicted_label[3]
        )

        if brand_of_true_label == brand_of_predicted_label:
            if model_of_true_label == model_of_predicted_label:
                if gen_of_true_label != gen_of_predicted_label:
                    gen += 1
            else:
                model += 1
        else:
            brand += 1

    return (
        gen,
        model,
        brand,
        (gen / dict_size),
        (model / dict_size),
        (brand / dict_size),
        dict_size,
    )
