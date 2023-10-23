# # Loading and Importing all required modules for the task

import numpy as np #Library for matrix operations
import pandas as pd #Library for data manipulation
import os #Library for directory operations
import skimage #Library for image processing
import warnings
warnings.filterwarnings('ignore')

#The KNN class that we will use to classify the images, which is also in the same directory
from KNN_Classifier import *

#The following are the metrics that we will use to evaluate our model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Declaring all the paths(directories) that will be used in the task
Path_Dataset = '../Dataset/TB_Healthy_Pneumonia/' # Path to the dataset
Path_Dataset_Train = Path_Dataset + 'train/' # Path to the training set
Path_Dataset_Test = Path_Dataset + 'test/' # Path to the test set


# # Initializing dataframes to save the data
Dataframe_Dataset_Train = pd.DataFrame(columns=['File Name','Label','Path','Data']) # Dataframe to store Train data
Dataframe_Dataset_Test = pd.DataFrame(columns=['File Name','Label','Path','Data']) # Dataframe to store Test data


# # Functions that will be used for the preprocessing
def Image_Noise_Reduction(image):
    '''Function to reduce noise in the image'''
    blurred_image = skimage.filters.gaussian(image, sigma=0.5)
    return blurred_image

def Image_Enhance_Contrats(image):
    '''Function to enhance the contrast of the image'''
    enhanced_image = skimage.exposure.equalize_hist(image)
    return enhanced_image

def Image_Normalize(image):
    '''Function to normalize the image'''
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

def Image_Preprocessing(image):
    '''Function to preprocess the image'''
    blurred_image = Image_Noise_Reduction(image)
    enhanced_image = Image_Enhance_Contrats(blurred_image)
    normalized_image = Image_Normalize(enhanced_image)
    return np.array(normalized_image)


# # Loading, reading and saving the data in the dataframes
for Label in os.listdir(Path_Dataset_Train): #Iterates over the labels(Healthy,TB,Pneumonia) in the training dataset
    Path_Dataset_Train_Label = os.path.join(Path_Dataset_Train, Label)
    for File in os.listdir(Path_Dataset_Train_Label): #Iterates over the files in the label folder
        Path_Dataset_Train_File = os.path.join(Path_Dataset_Train_Label, File) #Path of the file
        Image_Data = np.array(skimage.transform.resize(skimage.io.imread(Path_Dataset_Train_File,as_gray=True), (100,100))).flatten() #Reads the image, resizes it to 100x100 and flattens it
        Preprocessed_Image = Image_Preprocessing(Image_Data) #Preprocesses the image
        Dataframe_Dataset_Train = Dataframe_Dataset_Train.append({'File Name':File, 'Label': Label, 'Path': Path_Dataset_Train_File, 'Data':Preprocessed_Image}, ignore_index=True)
print(Dataframe_Dataset_Train)


for Label in os.listdir(Path_Dataset_Test): #Iterates over the labels(Healthy,TB,Pneumonia) in the training dataset
    Path_Dataset_Test_Label = os.path.join(Path_Dataset_Test, Label)
    for File in os.listdir(Path_Dataset_Test_Label): #Iterates over the files in the label folder
        Path_Dataset_Test_File = os.path.join(Path_Dataset_Test_Label, File) #Path of the file
        Image_Data = np.array(skimage.transform.resize(skimage.io.imread(Path_Dataset_Test_File,as_gray=True), (100,100))).flatten() #Reads the image and converts it into a 100x100 grayscale image
        Preprocessed_Image = Image_Preprocessing(Image_Data) #Preprocesses the image
        Dataframe_Dataset_Test = Dataframe_Dataset_Test.append({'File Name':File, 'Label': Label, 'Path': Path_Dataset_Test_File, 'Data':Preprocessed_Image}, ignore_index=True)
print(Dataframe_Dataset_Test)        

# # Splitting the data into train and test sets
xtrain, ytrain = Dataframe_Dataset_Train['Data'].values, Dataframe_Dataset_Train['Label'].values
print(xtrain.shape, ytrain.shape)

xtest, ytest = Dataframe_Dataset_Test['Data'].values, Dataframe_Dataset_Test['Label'].values
print(xtest.shape, ytest.shape)

New_Xtrain = []
New_Xtest = []

for x in xtrain:
    New_Xtrain.append(x)

for x in xtest:
    New_Xtest.append(x)

New_Xtrain = np.array(New_Xtrain)
New_Xtest = np.array(New_Xtest)

print(np.shape(New_Xtrain))
print(np.shape(New_Xtest))


# # Below statetments are used to save the loaded data as well as read the data. 
np.save('../Saved Datasets/xtrain.npy', New_Xtrain)
np.save('../Saved Datasets/xtest.npy', New_Xtest)
np.save('../Saved Datasets/ytrain.npy', ytrain)
np.save('../Saved Datasets/ytest.npy', ytest)

xtrain = np.load('../Saved Datasets/xtrain.npy',allow_pickle=True)
ytrain = np.load('../Saved Datasets/ytrain.npy',allow_pickle=True)
xtest = np.load('../Saved Datasets/xtest.npy',allow_pickle=True)
ytest = np.load('../Saved Datasets/ytest.npy',allow_pickle=True)


# # Model Training, Hyperparameter optimization and evaluation
def Save_Model(model,path):
    '''Function to save the model for use in GUI(streamlit)'''
    import pickle as pkl
    pkl.dump(model, open(path, 'wb'))


List_Acc, List_Prec, List_Recall, List_F1 = [], [], [], []
Params = np.arange(3,13,2)
for n_neighbors in Params:
    knn = KNearestNeighbor(n_neighbors)
    knn.train(xtrain,ytrain)
    pred,confidence = knn.predict(xtest)

    Accuracy = accuracy_score(ytest,pred)*100
    Precision = precision_score(ytest,pred,average='macro')*100
    Recall = recall_score(ytest,pred,average='macro')*100
    F1 = f1_score(ytest,pred,average='macro')*100

    print(f"n_neighbors = {n_neighbors}")
    print(f"Accuracy = {Accuracy}")
    print(f"Precision = {Precision}")
    print(f"Recall = {Recall}")
    print(f"F1 = {F1}")
    print()

    Save_Model(knn,f'../Saved Models/KNN_{n_neighbors}.pkl')

    List_Acc.append(Accuracy); List_Prec.append(Precision); List_Recall.append(Recall); List_F1.append(F1)

Best_N_Neighbor = np.argmax(List_Acc)
print(f'The best parameter for n_neighbors is: {Params[Best_N_Neighbor]}')
