from C3D_model import C3Dnet
import numpy as np


import matplotlib
matplotlib.use('Agg')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    model =C3Dnet(487, (16, 112, 112, 3))
    model.summary()
    try:
        model.load_weights('models/C3D_Sport1M_weights_keras_2.2.4.h5')
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)

    # 16 black frames with 3 channels

    print("[Info] Loading labels...")
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    print("[Info] Loading a sample video...")
    cap = cv2.VideoCapture('dM06AMFLsrc.mp4')

    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)

    start_frame = 2000
    X = vid[start_frame:(start_frame + 16), :, :, :]

    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    X -= mean_cube

    # center crop
    X = X[:, 8:120, 30:142, :] # (l, h, w, c)

    X = np.expand_dims(X, axis=0)

    prediction_softmax = model.predict(X)
    predicted_class = np.argmax(prediction_softmax)

    print('Success, predicted class is: {}'.format(labels[predicted_class]))

    top_inds = prediction_softmax[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5 probabilities and labels:')
    for i in top_inds:
        print('{1}: {0:.5f}'.format(prediction_softmax[0][i], labels[i]))

if __name__ == "__main__":
    main()
