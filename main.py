import argparse
import os
import os.path as osp

import cv2
from tqdm import tqdm
from model.classify import PassportClassifier


def save_img(save_dir, img_name, img):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir,img_name), img)

def main(config):

    classifier = PassportClassifier(pth_device='cpu')

    if os.path.isfile(config.images_path):
        img_paths = [config.images_path]
    else:
        img_paths = [os.path.join(config.images_path, f)
                        for f in os.listdir(config.images_path)]
    res = {}
    for img_path in img_paths:
        img = cv2.imread(img_path)
        
        pred = classifier.classify(img)
        print(f'{img_path} ---> {pred}')
        if pred in res:
            res[pred] += 1
        else:
            res[pred] = 1

    print(res)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path', type=str,
                        help='path of input image or directory of images')
    
    args = parser.parse_args()
    main(args)
