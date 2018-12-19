from sklearn.cluster import KMeans
import pickle
import gzip
from PIL import Image
import os
import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
import tensorflow as tf
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.contrib.tensor_forest.python import tensor_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


MNISTTrainingData       = []
MNISTTrainingTarget     = []
MNISTValidationData     = []
MNISTValidationTarget   = []
MNISTTestingData        = []
MNISTTestingTarget      = []
USPSTestingData         = []
USPSTestingTarget       = []

# Reading MNIST Data from the given file.
def readMNISTData(FileName):

    f = gzip.open(FileName, 'rb')
    TrainingData, ValidationData, TestingData = pickle.load(f, encoding='latin1')
    f.close()

    return TrainingData[0], TrainingData[1], ValidationData[0], ValidationData[1], TestingData[0], TestingData[1]

# Reading USPS data to test our different classifiers.
def readUSPSData(FileName):

    USPSMat  = []
    USPSTar  = []
    curPath  = FileName
    savedImg = []

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)

    return USPSMat, USPSTar

# Code to plot confusion matrix using Predicted and Actual test targets.
def ConfusionMatrix(Predicted, Actual, Label):

    Confusion_Matrix = np.zeros(shape=(10,10))

    for i in range(Predicted.shape[0]):

        Actual_Value = Actual[i]
        Predicted_Value = Predicted[i]

        Confusion_Matrix[Actual_Value][Predicted_Value] = Confusion_Matrix[Actual_Value][Predicted_Value] + 1

    # Plotting confusion matrix.
    Target = [0,1,2,3,4,5,6,7,8,9]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(Target))
    plt.xticks(tick_marks, Target)
    plt.yticks(tick_marks, Target)

    sns.heatmap(pd.DataFrame(Confusion_Matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    fig.canvas.set_window_title('Confusion matrix: ' + Label)

    plt.show()


# This function outputs Y for a particular value of x,w (Linear Regression). 
def GetValTest(Data_PHI, W):
    Y = np.dot(Data_PHI, W)
    return Y

# Finding the accuracy for the predicted values of Logistic Regression Classifier.
def Accuracy(Predicted, Target):

    Actual    = np.argmax(Target, axis=1)

    diff = np.array(Predicted) - np.array(Actual)
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

# Code to compute softmax of the obtained output Y. By using softmax we are finding the probablities of 
# whether the input belongs to a particular class.
def GetSoftmax(Y):

    return (np.exp(Y.T) / np.sum(np.exp(Y), axis=1)).T

# To find the predicted class label from the softmax output. We take the max probability from the softmax 
# output to be the predicted class for the input x.
def ToClassLabel(Softmax):
    
    return Softmax.argmax(axis=1)

# Calculating the total cost/loss of the predicted target. By doing this we find the 
# error of the predicted values.
def Cost(Softmax, Target):

    return np.mean(-np.sum(np.log(Softmax) * (Target), axis=1))


# ------------------------- Logistic Regression using SGD Optimization --------------------------- #

def SGD_LogisticRegression(MNISTTrainingData, MNISTValidationData, MNISTTestingData, MNISTTrainingTarget, MNISTValidationTarget, MNISTTestingTarget, USPSTestingData, USPSTestingTarget, Lamda, LearningRate):

    W = np.zeros((784, 10))
    W_Now = np.add(1, W)

    ValCost   = []
    TRCost    = []
    TestCost  = []
    TRAccuracy = []
    ValAccuracy = []
    MNISTTestAccuracy = []
    USPSTestAccuracy  = []

    for i in range(0, 500):

        #-----------------TrainingData Accuracy---------------------#
        TR_OUT = GetValTest(MNISTTrainingData, W_Now) 
        Softmax_Tr = GetSoftmax(TR_OUT)
        Tr_Cost  = Cost(Softmax_Tr, MNISTTrainingTarget)
        TRCost.append(float(Tr_Cost))
        Tr_Predicted = ToClassLabel(Softmax_Tr)
        Acc = Accuracy(Tr_Predicted, MNISTTrainingTarget)
        TRAccuracy.append(Acc)

        #-----------------ValidationData Accuracy---------------------#
        VAL_OUT = GetValTest(MNISTValidationData, W_Now) 
        Softmax_Val = GetSoftmax(VAL_OUT)
        Val_Cost  = Cost(Softmax_Val, MNISTValidationTarget)
        ValCost.append(float(Val_Cost))
        Val_Predicted = ToClassLabel(Softmax_Val)
        Acc = Accuracy(Val_Predicted, MNISTValidationTarget)
        ValAccuracy.append(Acc)

        #-----------------TestingData Accuracy MNIST------------------------#
        TEST_OUT = GetValTest(MNISTTestingData, W_Now) 
        Softmax_Test = GetSoftmax(TEST_OUT)
        Test_Cost  = Cost(Softmax_Test, MNISTTestingTarget)
        TestCost.append(float(Test_Cost))
        MNISTTest_Predicted = ToClassLabel(Softmax_Test)
        Acc = Accuracy(MNISTTest_Predicted, MNISTTestingTarget)
        MNISTTestAccuracy.append(Acc)

        #-----------------TestingData Accuracy USPS------------------------#
        TEST_OUT = GetValTest(USPSTestingData, W_Now) 
        Softmax_Test = GetSoftmax(TEST_OUT)
        Test_Cost  = Cost(Softmax_Test, USPSTestingTarget)
        TestCost.append(float(Test_Cost))
        USPSTest_Predicted = ToClassLabel(Softmax_Test)
        Acc = Accuracy(USPSTest_Predicted, USPSTestingTarget)
        USPSTestAccuracy.append(Acc)

        # Updating Weights at each iteration
        x = np.transpose(MNISTTrainingTarget[i] - TR_OUT[i])
        x = np.transpose(x.reshape(x.shape[0], -1))
        z = MNISTTrainingData[i]
        z = z.reshape(z.shape[0], -1)
        Delta_E_D     = -np.dot(z, x)
        La_Delta_E_W  = np.dot(Lamda, W_Now)
        Delta_E       = np.add(Delta_E_D, La_Delta_E_W)

        Delta_W       = -np.dot(LearningRate, Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next


    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('---------------------------------------------------------')
    print ('---------------Logistic Regression-----------------------')
    print ('---------------------------------------------------------')
    print ("Lambda  = "+ str(Lamda) + "\nLearning Rate = " + str(LearningRate))
    print ("Training Accuracy   = " + str(max(TRAccuracy)*100))
    print ("Validation Accuracy = " + str(max(ValAccuracy)*100))
    print ("Testing Accuracy (MNIST)  = " + str(max(MNISTTestAccuracy)*100))
    print ("Testing Accuracy (USPS)   = " + str(max(USPSTestAccuracy)*100))

    return MNISTTest_Predicted, USPSTest_Predicted

# Logistic Regression main method.
def LogisticRegression(Lambda, LearningRate):

    MNISTPredictedTest, USPSPredictedTest = SGD_LogisticRegression(MNISTTrainingData, 
                            MNISTValidationData, MNISTTestingData, MNISTTrainingTargetCategorical, 
                            MNISTValidationTargetCategorical, MNISTTestingTargetCategorical, 
                            USPSTestingData, USPSTestingTargetCategorical, Lambda, LearningRate)

    ConfusionMatrix(MNISTPredictedTest, MNISTTestingTarget, 'Logistic Regression (MNIST)')
    ConfusionMatrix(USPSPredictedTest, USPSTestingTarget, 'Logistic Regression (USPS)')

    return MNISTPredictedTest, USPSPredictedTest


# ------------------------------ Neural Network Implementation -------------------------------- #

# Tensorflow Model

def NeuralNetwork(LearningRate, HiddenLayers, Epochs, BatchSize):

    # Defining Placeholder
    # Placeholders are like variables which are assigned data at a later date.
    # By creating placeholders, we only assign memory(optional) where data is stored later on.
    InputTensor  = tf.placeholder(tf.float32, [None, len(MNISTTrainingData[0])])
    OutputTensor = tf.placeholder(tf.float32, [None, 10])

    # Initializing the weights to Normal Distribution
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.01))

    # Initializing the input to hidden layer weights
    InputLayerWeights  = init_weights([len(MNISTTrainingData[0]), HiddenLayers])
    # Initializing the hidden to output layer weights
    OutputLayerWeights = init_weights([HiddenLayers, 10])   

    # Computing values at the hidden layer
    # relu to convert to linear data
    HiddenLayer = tf.nn.relu(tf.matmul(InputTensor, InputLayerWeights))
    # Computing values at the output layer
    OutputLayer = tf.matmul(HiddenLayer, OutputLayerWeights)
    
    # Defining Error Function
    # It computes inaccuracy of predictions in classification problems.
    ErrorFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = OutputLayer, labels = OutputTensor))
    # Defining Learning Algorithm and Training Parameters.
    Training = tf.train.GradientDescentOptimizer(LearningRate).minimize(ErrorFunction)

    # Prediction Function.
    Prediction = tf.argmax(OutputLayer, 1)

    # Training the Model.
    TrainingAccuracy = []
    ValidationAccuracy = []

    # TensorFlow session is an object where all operations are run.
    with tf.Session() as Session:

        tf.global_variables_initializer().run()

        # Iterating for the number of Epochs.
        for Epoch in tqdm_notebook(range(Epochs)):

            p = np.random.permutation(range(len(MNISTTrainingData)))

            TrainingD = MNISTTrainingData[p]
            TrainingT = MNISTTrainingTargetCategorical[p]

            for Start in range(0, len(TrainingD), BatchSize):
                End = Start + BatchSize
                Session.run(Training, feed_dict = { InputTensor: TrainingD[Start:End], 
                                          OutputTensor: TrainingT[Start:End] })

            TrainingAccuracy.append(np.mean(np.argmax(TrainingT, axis=1) ==
                             Session.run(Prediction, feed_dict={InputTensor: TrainingD,
                                                             OutputTensor: TrainingT})))

            ValidationAccuracy.append(np.mean(np.argmax(MNISTValidationTargetCategorical, axis=1) ==
                             Session.run(Prediction, feed_dict={InputTensor: MNISTValidationData,
                                                    OutputTensor: MNISTValidationTargetCategorical})))

        MNISTPredictedTestTarget = Session.run(Prediction, feed_dict={InputTensor: MNISTTestingData})

        USPSPredictedTestTarget = Session.run(Prediction, feed_dict={InputTensor: USPSTestingData})
    
    MNISTRight = 0
    MNISTWrong = 0
    for i,j in zip(MNISTTestingTarget, MNISTPredictedTestTarget):
        
        if i == j:
            MNISTRight = MNISTRight + 1
        else:
            MNISTWrong = MNISTWrong + 1

    USPSRight = 0
    USPSWrong = 0
    for i,j in zip(USPSTestingTarget, USPSPredictedTestTarget):
        
        if i == j:
            USPSRight = USPSRight + 1
        else:
            USPSWrong = USPSWrong + 1

    ConfusionMatrix(MNISTPredictedTestTarget, MNISTTestingTarget, 'Neural Network (MNIST)')
    ConfusionMatrix(USPSPredictedTestTarget, USPSTestingTarget, 'Neural Network (USPS)')

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('--------------------------------------------------')
    print ('-------------Neural Network-----------------------')
    print ('--------------------------------------------------')
    print("Training Accuracy: " + str(max(TrainingAccuracy)*100))
    print("Validation Accuracy: " + str(max(ValidationAccuracy)*100))
    print("Errors: " + str(MNISTWrong), " Correct :" + str(MNISTRight))
    print("Testing Accuracy (MNIST): " + str(MNISTRight / (MNISTRight + MNISTWrong)*100))
    print("Errors: " + str(USPSWrong), " Correct :" + str(USPSRight))
    print("Testing Accuracy (USPS): " + str(USPSRight / (USPSRight + USPSWrong)*100))

    return MNISTPredictedTestTarget, USPSPredictedTestTarget

def RandomForest(Estimators):

    Classifier = RandomForestClassifier(n_estimators = Estimators)
    Classifier.fit(MNISTTrainingData, MNISTTrainingTarget)

    ValidationAccuracy = Classifier.score(MNISTValidationData, MNISTValidationTarget)

    MNISTTestingAccuracy = Classifier.score(MNISTTestingData, MNISTTestingTarget)
    USPSTestingAccuracy = Classifier.score(USPSTestingData, USPSTestingTarget)

    MNISTPredictedTestTarget = Classifier.predict(MNISTTestingData)
    USPSPredictedTestTarget = Classifier.predict(USPSTestingData)

    ConfusionMatrix(MNISTPredictedTestTarget, MNISTTestingTarget, 'Random Forest (MNIST)')
    ConfusionMatrix(USPSPredictedTestTarget, USPSTestingTarget, 'Random Forest (USPS)')

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('--------------------------------------------------')
    print ('-------------Random Forest------------------------')
    print ('--------------------------------------------------')
    print("Validation Accuracy: " + str(ValidationAccuracy*100))
    print("Testing Accuracy (MNIST): " + str(MNISTTestingAccuracy*100))
    print("Testing Accuracy (USPS): " + str(USPSTestingAccuracy*100))

    return MNISTPredictedTestTarget, USPSPredictedTestTarget

def SVM(Kernel):

    Classifier = SVC(kernel=Kernel, C=2, gamma=0.05)
    Classifier.fit(MNISTTrainingData, MNISTTrainingTarget)

    ValidationAccuracy = Classifier.score(MNISTValidationData, MNISTValidationTarget)
    MNISTTestingAccuracy = Classifier.score(MNISTTestingData, MNISTTestingTarget)
    USPSTestingAccuracy = Classifier.score(USPSTestingData, USPSTestingTarget)

    MNISTPredictedTestTarget = Classifier.predict(MNISTTestingData)
    USPSPredictedTestTarget  = Classifier.predict(USPSTestingData)

    ConfusionMatrix(MNISTPredictedTestTarget, MNISTTestingTarget, 'SVM (MNIST)')
    ConfusionMatrix(USPSPredictedTestTarget, USPSTestingTarget, 'SVM (USPS)')

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('--------------------------------------------------')
    print ('--------------------SVM---------------------------')
    print ('--------------------------------------------------')
    print("Validation Accuracy: " + str(ValidationAccuracy*100))
    print("Testing Accuracy (MNIST): " + str(MNISTTestingAccuracy*100))
    print("Testing Accuracy (USPS): " + str(USPSTestingAccuracy*100))

    return MNISTPredictedTestTarget, USPSPredictedTestTarget

def MajorityVoting(PredictedTestTarget_LR, PredictedTestTarget_NN, PredictedTestTarget_RF, 
                        PredictedTestTarget_SVM, Label):

    TestTarget = np.zeros(len(PredictedTestTarget_LR), dtype=int)
    for i in range(0, len(PredictedTestTarget_LR)):
        l = np.zeros((4,), dtype=int)
        l[0] = PredictedTestTarget_LR[i]
        l[1] = PredictedTestTarget_NN[i]
        l[2] = PredictedTestTarget_RF[i]
        l[3] = PredictedTestTarget_SVM[i]

        Counts = np.bincount(l)
        TestTarget[i] = np.argmax(Counts)

    if(Label == 'MNIST'):
        TestingAccuracy = accuracy_score(MNISTTestingTarget, TestTarget)
    else:
        TestingAccuracy = accuracy_score(USPSTestingTarget, TestTarget)

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('--------------------------------------------------')
    print ('--------------Majority Voting (' + Label + ')----------------')
    print ('--------------------------------------------------')
    print("Testing Accuracy: " + str(TestingAccuracy*100))

    
print("Reading MNIST Data .............................................................")
MNISTTrainingData, MNISTTrainingTarget, MNISTValidationData, MNISTValidationTarget, MNISTTestingData, MNISTTestingTarget = readMNISTData('mnist.pkl.gz')
print("Reading of MNIST Data Completed")
print("Reading USPS Data ..............................................................")
USPSTestingData, USPSTestingTarget = readUSPSData('USPSdata/Numerals')
print("Reading of USPS Data Completed")


MNISTTrainingTargetCategorical     = np_utils.to_categorical(MNISTTrainingTarget)
MNISTValidationTargetCategorical   = np_utils.to_categorical(MNISTValidationTarget)
MNISTTestingTargetCategorical      = np_utils.to_categorical(MNISTTestingTarget)
USPSTestingTargetCategorical       = np_utils.to_categorical(USPSTestingTarget)


print("This may take a while............................................................")
print("\nRunning Logistic Regression Classifier.........................................")
MNISTPredictedTestTarget_LR, USPSPredictedTestTarget_LR = LogisticRegression(0.1, 0.003)

print("\nRunning Neural Network Classifier..............................................")
MNISTPredictedTestTarget_NN, USPSPredictedTestTarget_NN  = NeuralNetwork(1.0, 100, 50, 128)

print("\nRunning Random Forest Classifier...............................................")
MNISTPredictedTestTarget_RF, USPSPredictedTestTarget_RF  = RandomForest(100)

print("\nRunning SVM Classifer..........................................................")
print("This may take a while (Approx 30 min)..............................................")
MNISTPredictedTestTarget_SVM, USPSPredictedTestTarget_SVM = SVM('rbf')

# Ensemble Classifier using majority voting.
print("\nRunning Ensemble Classifer (Majority Vaoting).................................")
MajorityVoting(MNISTPredictedTestTarget_LR, MNISTPredictedTestTarget_NN, 
                    MNISTPredictedTestTarget_RF, MNISTPredictedTestTarget_SVM, 'MNIST')
MajorityVoting(USPSPredictedTestTarget_LR, USPSPredictedTestTarget_NN, 
                    USPSPredictedTestTarget_RF, USPSPredictedTestTarget_SVM, 'USPS')



