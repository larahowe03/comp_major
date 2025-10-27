import os
import cv2
import sklearn

xdata = []
ydata = []

for piece in os.listdir("chess-dataset"):
    for image in os.listdir(os.path.join("chess-dataset", piece)):
        src = os.path.join("chess-dataset", piece, image)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        ydata.append(piece)
        xdata.append(img)

RANDOM_SEED = 7
    
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
    xdata, ydata, test_size=0.3, random_state=RANDOM_SEED, stratify=ydata
)

print(len(xtrain), len(xtest), len(ytrain), len(ytest))