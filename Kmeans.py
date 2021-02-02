import numpy as np
import loadMNIST
from os.path import join
import os
import sys
import copy
import time
import matplotlib.pyplot as plt


def show_centroid(labels_centroids):
    cols = 5
    rows = 2
    plt.figure(figsize=(30, 20))
    index = 1
    for label in labels_centroids:
        image = labels_centroids[label]
        title_text = label
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


def KMeans(k, data):
    # initialize centroids randomly
    old_centroids, centroids = init_centroids_random(k, len(data[0]))
    # initialize centroids based on examples
    # old_centroids, centroids = init_centroids_examples(k, data)
    clusters_list = init_clusters(k)
    iter_num = 0
    while not shouldStop(centroids, old_centroids, iter_num):
        iter_num += 1
        print("iteration number: ", iter_num)
        old_centroids = copy.deepcopy(centroids)
        clusters_list = copy.deepcopy(init_clusters(k))
        # classify images to clusters by finding the minimal norm of the image vector minus the centroid vector
        for img_index in range(0, len(data)):
            min_distance = sys.maxsize
            min_centroid_index = -1
            for curr_centroid_index in range(0, len(old_centroids)):
                curr_distance = np.linalg.norm(data[img_index] - old_centroids[curr_centroid_index])
                if min_distance > curr_distance:
                    min_distance = curr_distance
                    min_centroid_index = curr_centroid_index
            clusters_list[min_centroid_index].append(img_index)
        # generate new centroids by calculating the average for each cluster
        for cluster_index in range(0, len(clusters_list)):
            if len(clusters_list[cluster_index]) > 0:
                sum_matrix = calc_sum_matrix(clusters_list[cluster_index], data)
                avg_matrix = (float(1. / len(clusters_list[cluster_index]))) * sum_matrix
                centroids[cluster_index] = copy.deepcopy(avg_matrix)
    return centroids, clusters_list


def shouldStop(centroids, old_centroids, iterations):
    if iterations >= 100:
        return True
    for index in range(0, len(centroids)):
        distance = np.linalg.norm(centroids[index] - old_centroids[index])
        print(distance)
        if distance > 0:  # != 0:
            return False
    return True


def init_centroids_random(k, size):
    old_centroids = []
    centroids = []
    for index in range(0, k):
        centroids.append(np.random.rand(size, size))
        old_centroids.append(np.zeros((size, size)))
    return old_centroids, centroids


def init_centroids_examples(k, data):
    old_centroids = []
    centroids = [data[56], data[102], data[25], data[281], data[142], data[543], data[62], data[103], #5 - 332
                 data[144], data[183]]
    for index in range(0, k):
        old_centroids.append(np.zeros((len(data[0]), len(data[0]))))
    return old_centroids, centroids


def init_clusters(k):
    clusters = []
    for index in range(0, k):
        clusters.append([])
    return clusters


def calc_sum_matrix(cluster, data):
    sum_matrix = np.zeros((len(data[0]), len(data[0])))
    for img_index in cluster:
        sum_matrix = sum_matrix + data[img_index]
    return sum_matrix


def classify_centroids(centroids, clusters, labels):
    classification = {}
    labels_count = {}
    k = len(centroids)
    for centroid_index in range(0, k):
        labels_count[centroid_index] = np.zeros(k)
        for img_index in clusters[centroid_index]:
            img_label = labels[img_index]
            labels_count[centroid_index][img_label] += 1
        print(labels_count[centroid_index], " total in this cluster: ", sum(labels_count[centroid_index]))
    for digit in range(0, k):
        max_centroid_index = -1
        max_val = 0
        for centroid_index in labels_count:
            if sum(labels_count[centroid_index]) != 0:
                curr_val = (labels_count[centroid_index][digit] / sum(labels_count[centroid_index]))
            else:
                curr_val = 0
            if curr_val >= max_val:
                max_val = curr_val
                max_centroid_index = centroid_index
        classification[digit] = centroids[max_centroid_index]
        labels_count.pop(max_centroid_index)
    return classification


def test_classification(labels_centroids, data):
    classification = {}
    for img_index in range(0, len(data)):
        min_distance = sys.maxsize
        min_centroid = -1
        for label in labels_centroids:
            distance = np.linalg.norm(data[img_index] - labels_centroids[label])
            if distance < min_distance:
                min_centroid = label
                min_distance = distance
        classification[img_index] = min_centroid
    return classification


def calc_success_rate(true_labels, predicted_labels):
    suc_labeling = 0
    for img_index in range(0, len(true_labels)):
        if true_labels[img_index] == predicted_labels[img_index]:
            suc_labeling += 1
    suc_percents = suc_labeling / len(true_labels)
    return suc_percents


# ======= Main program =======
# read MNIST data
cwd = os.getcwd()
input_path = cwd + '\\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')
mnist_dataloader = loadMNIST.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                             test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# represent as matrices
img_list = []
for i in range(0, len(x_train)):
    img_list.append(np.asarray(x_train[i]))

# normalize values to values between 0-1
img_list_normalized = []
for i in range(0, len(img_list)):
    img_list_normalized.append(float(1 / 255) * img_list[i])

# run K-Means algorithm and classify the centroids
start_time_training = time.time()
final_centroids, final_clusters = KMeans(10, img_list_normalized)
labels_to_centroids = classify_centroids(final_centroids, final_clusters, y_train)
show_centroid(labels_to_centroids)

# run classification test
test_list = []
for i in range(0, len(x_test)):
    test_list.append(np.asarray(x_test[i]))
test_list_normalized = []
for i in range(0, len(test_list)):
    test_list_normalized.append((float(1. / 255.)) * test_list[i])
test_to_labels = test_classification(labels_to_centroids, test_list_normalized)
success_rate = calc_success_rate(y_test, test_to_labels)

print("-----Success rate is: ", success_rate * 100, "%")