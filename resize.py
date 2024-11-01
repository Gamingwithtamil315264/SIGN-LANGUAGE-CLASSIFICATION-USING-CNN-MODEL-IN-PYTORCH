import cv2
import os

def resize_images(folder_path, new_width, new_height):
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for f in folders:
        d=os.path.join('data',f)
        for filename in os.listdir(d):
            print(filename)
            img_path = os.path.join(d, filename)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, (new_width, new_height))
                print("c")
                cv2.imwrite(img_path, resized_img) 

if __name__ == "__main__":
    folder_path = "./data/"  # Replace with your folder path
    new_width = 32
    new_height = 32

    resize_images(folder_path, new_width, new_height)
