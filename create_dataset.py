import cv2
import os
import random
import shutil
from dataHandler import augment


def start_capture(name):
    num_of_images = 0
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        # os.makedirs(path)
        train_path = "./dataset/train/" + name + "/"
        photo_path = "./dataset/photos/" + name + "/"
        test_path = "./dataset/test/" + name + "/"
        os.makedirs(photo_path)
        os.makedirs(train_path)
        os.makedirs(test_path)

    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:

        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255))
            new_img = img[y:y + h, x:x + w]
        cv2.imshow("FaceDetection", img)
        key = cv2.waitKey(1) & 0xFF

        try:
            cv2.imwrite((train_path + "/" + str(num_of_images) + name + ".jpg"), new_img)
            cv2.imwrite((photo_path + "/" + str(num_of_images) + name + ".jpg"), new_img)
            num_of_images += 1
        except:

            pass
        if key == ord("q") or key == 27 or num_of_images > 50:
            break
    cv2.destroyAllWindows()
    return num_of_images


def move(name):
    train_folder = os.path.join('dataset', 'train', name)
    test_folder = os.path.join('dataset', 'test', name)

    # Get the list of image filenames in the train folder
    image_filenames = os.listdir(train_folder)

    # Calculate the number of images to move to the test folder
    num_images_to_move = int(len(image_filenames) * 0.2)

    # Randomly select which images to move
    images_to_move = random.sample(image_filenames, num_images_to_move)

    # Move the selected images from the train folder to the test folder
    for image_filename in images_to_move:
        src_path = os.path.join(train_folder, image_filename)
        dst_path = os.path.join(test_folder, image_filename)
        shutil.move(src_path, dst_path)
