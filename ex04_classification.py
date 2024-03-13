# -*- coding: utf-8 -*-
"""
Spectral channels (visual bands)
1 0.45-0.52 µm, blue-green
2 0.52-0.60 µm, green
3 0.63-0.69 µm, red
4 0.76-0.90 µm, near infrared
5 1.55-1.75 µm, mid infrared
6 10.4-12.5 µm (60 × 60 m) Thermal channel
7 2.08-2.35 µm, mid infrared

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
# from mlxtend.plotting import plot_decision_regions

plt.close('all')

label_to_color = {
    1: [0,  0, 128], # water
    2: [0,  128, 0], # forest
    3: [0,  255,  0], # vegetation
    4: [0, 221, 221], # ice
    5: [255, 255, 255], # snow
    6: [255, 0, 0], # rock
    7: [80, 80, 80] # shadow
}

# convert one channel label image [nxm] to a given colormap 
# resutling in a rgb image [nxmx3]
def label2rgb(img_label, label_to_color):
    h, w = img_label.shape
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for gray, rgb in label_to_color.items():
        img_rgb[img_label == gray, :] = rgb
    return img_rgb

# options
num_channels = 7 # 3 or 7

# load label image (values from 1 to 7 in a single image)
img_label_train = plt.imread('data/labels_train.tif')
img_label_train_color = label2rgb(img_label_train,label_to_color)

# load traindata (7 images of 1 band each)
traindata = np.zeros((img_label_train.shape[0],img_label_train.shape[1],7))
for ii in range(0,7):
    I = plt.imread('data/band' + str(ii+1) + '_train.tif')
    traindata[:,:,ii] = I

# flatten the train data and labels
if num_channels == 7:
    X_train = traindata.reshape(traindata.shape[0]*traindata.shape[1],num_channels) # 1000x500x7
elif num_channels == 3:
    X_train = traindata[:,:,:3].reshape(traindata.shape[0]*traindata.shape[1],num_channels) # 1000x500x3 
else:
    raise ValueError
    
y_train = img_label_train.reshape(img_label_train.shape[0]*img_label_train.shape[1]) # for supervised learning; 1000x500

# perform Gaussian Naïve Bayes classification 
clf = GaussianNB() # define model
clf.fit(X_train, y_train) # training occurs here
y_train_prediction = clf.predict(X_train) # make prediction on the X_train dataset
y_train_reshape = y_train_prediction.reshape(traindata.shape[0],traindata.shape[1])
y_train_RGB = label2rgb(y_train_reshape,label_to_color)
# print(clf.score(X_train,y_train)*100) # same output as accuracy_score
# #Plot decision region:
# plt.figure
# plot_decision_regions(X_train,y_train,clf=clf,legend=1)
# #Adding axes annotations:
# plt.xlabel('X_train')
# plt.ylabel('y_train')
# plt.title('Gaussian Naive Bayes') 
# plt.show()

fig, axes = plt.subplots(1,2)
axes[0].imshow(y_train_RGB) # restore array size
axes[0].set_title('predicted (train) image')
axes[1].imshow(img_label_train_color)
axes[1].set_title('label (train) image')
plt.tight_layout()
plt.savefig('prediction_vs_trainlabel_%sChannels.png' % num_channels)

train_accuracy = accuracy_score(y_train, y_train_prediction)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')


# load label test images (values from 1 to 7 in a single image)
img_label_test = plt.imread('data/labels_test.tif')
img_label_test_color = label2rgb(img_label_test,label_to_color)

# load testdata (7 images of 1 band each)
testdata = np.zeros((img_label_test.shape[0],img_label_test.shape[1],7))
for ii in range(0,7):
    I = plt.imread('data/band' + str(ii+1) + '_test.tif')
    testdata[:,:,ii] = I
    
# run prediction on test data
if num_channels == 7:
    X_test = testdata.reshape(testdata.shape[0]*testdata.shape[1],num_channels) # 872x632x7
elif num_channels == 3:
    X_test = testdata[:,:,:3].reshape(testdata.shape[0]*testdata.shape[1],num_channels) # 872x632x3 
else:
    raise ValueError

y_predicted = clf.predict(X_test)
y_predicted_reshape = y_predicted.reshape(testdata.shape[0],testdata.shape[1])
y_predicted_RGB = label2rgb(y_predicted_reshape,label_to_color)
fig, axes = plt.subplots(1,2)
axes[0].imshow(y_predicted_RGB)
axes[0].set_title('predicted (test) image')
axes[1].imshow(img_label_test_color)
axes[1].set_title('label (test) image')
plt.tight_layout()
plt.savefig('prediction_vs_testlabel_%sChannels.png' % num_channels)

# calculate quality matrices i.e. confusion matrix
y_test = img_label_test.reshape(img_label_test.shape[0]*img_label_test.shape[1])
C = confusion_matrix(y_test,y_predicted)
print('Confusion matrix = \n' + str(C))


# Extract TP, FP, TN, FN from the confusion matrix
# Initialize arrays to store Producer's Accuracy (PA) and User's Accuracy (UA) for each class
OA = np.sum(np.diagonal(C)) / np.sum(C) # overall value
print(f"Overall Accuracy = {OA * 100:.2f}%")

PA = np.zeros(7) # num_classes = 7
UA = np.zeros(7)
for ii in range(0,7):
    # True Positives, False Positives, True Negatives, False Negatives for class i
    TP = C[ii, ii]
    FP = np.sum(C[:, ii]) - TP
    FN = np.sum(C[ii, :]) - TP
    TN = np.sum(C) - TP - FP - FN

    # Calculate PA, UA for each class
    PA[ii] = C[ii, ii] / np.sum(C[ii,:]) #TP / (TP + FP) if (TP + FP) > 0 else 0
    UA[ii] = C[ii, ii] / np.sum(C[:,ii]) # TP / (TP + FN) if (TP + FN) > 0 else 0
      
for i in range(0,7):
    print(f"Class {i + 1}: Producer's Accuracy = {PA[i] * 100:.2f}%, User's Accuracy = {UA[i] * 100:.2f}%")


# calculate Kappa coefficient
po = np.sum(C) * np.sum(np.diagonal(C))
pc = np.sum(np.sum(C, axis=0) * np.sum(C, axis=1))
kappa = (po - pc) / ((np.sum(C) ** 2) - pc)
print(f"Kappa's Coefficient: {kappa:.4f}")

