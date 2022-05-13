from os.path import isfile, join
import cv2
import time
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import random

TRAINING_SET_FOLDER = "../training"
GENERATED_TRAINING_DATA_SET_FOLDER = "training/generated"
GENERATED_TESTING_DATA_SET_FOLDER = "testing/generated"
TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER = "training/training_images/taichi"
VOXCELEB_TRAINING_IMAGES_VIDEOS_SET_FOLDER = "training/training_images/voxceleb"
TAICHI_TESTING_IMAGES_VIDEOS_SET_FOLDER = "testing/testing_images/taichi"
VOXCELEB_TESTING_IMAGES_VIDEOS_SET_FOLDER = "testing/testing_images/voxceleb"
GENERATED_FRAMES_FOLDER = "/frames"
GENERATED_VIDEOS_FOLDER = "/video"
VIDEO_DATASET_FOLDER = "../dataset/videos"


def resize_video(path, filename, new_path):
    # create a new folder
    cap = cv2.VideoCapture(path + filename)
    new_path_file = os.makedirs(new_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(new_path_file + '1' + filename, fourcc, 5, (256, 256))

    if cap.isOpened() == False:
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, (256, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def generate_frames_from_video(source_path, output_path):
    print(source_path)
    print(output_path)
    try:
        os.mkdir(output_path)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(source_path)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_path + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break


def generate_video_from_frames(source_path, output_path, fps):
    try:
        os.mkdir(output_path)
    except OSError:
        pass
    image_array = []
    files = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
    files.sort(key=lambda x: int(float(x.split('.')[0])))
    for i in range(len(files)):
        img = cv2.imread(source_path + "/" + files[i])
        size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


def save_image_to_folder(path, name, image):
    try:
        os.mkdir(path)
    except OSError:
        pass
    print(path)
    cv2.imwrite(path + name, image)


def generate_frames_from_videos(folder):
    files = os.listdir(folder)
    for vid_inx in range(len(files)):
        generate_frames_from_video(folder + "/" + files[vid_inx], "../test_taichi" + "/" + files[vid_inx])


def get_images_from_folder(data_folder, index):
    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()
    images = []
    temp_images = []
    training_list = os.listdir(data_folder)
    training_list.remove('.DS_Store')
    files = os.listdir(data_folder + "/" + training_list[index])
    files.sort(key=lambda x: int(float(x.split('.')[0])))
    for file in files:
        if not file.startswith("."):
            img = transform(cv2.imread(data_folder + "/" + training_list[index] + "/" + file))
            images.append(img)
    return images


def get_dataset_size(data_folder):
    training_list = os.listdir(data_folder)
    training_list.remove('.DS_Store')
    return len(training_list)


def get_dataloader(frames, batch_no):
    data_loader = DataLoader(frames, batch_size=batch_no)
    return data_loader


def save_images_to_folder(frame, dataset, foldername, filename, is_test):
    existing_path = GENERATED_TESTING_DATA_SET_FOLDER if is_test == True else GENERATED_TRAINING_DATA_SET_FOLDER
    if os.path.exists(existing_path + '/' + dataset):
        if os.path.exists(existing_path + '/' + dataset + '/' + foldername):
            save_image(frame,
                       existing_path + '/' + dataset + '/' + foldername + '/' + '{}.jpg'.format(filename))
        else:
            os.makedirs(existing_path + '/' + dataset + '/' + foldername)
            save_image(frame,
                       existing_path + '/' + dataset + '/' + foldername + '/' + '{}.jpg'.format(filename))
    else:
        os.makedirs(existing_path + '/' + dataset + '/' + foldername)
        save_image(frame,
                   existing_path + '/' + dataset + '/' + foldername + '/' + '{}.jpg'.format(filename))


if __name__ == "__main__":
    print(get_dataset_size("../" + TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER))
    t = get_images_from_folder("../" + TAICHI_TRAINING_IMAGES_VIDEOS_SET_FOLDER, 1)
    src = t[0]
    des = t[1:]
    print(src.shape)
    print(len(des))
