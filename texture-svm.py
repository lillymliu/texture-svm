import cv2
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import random
import time
import itertools

class SVM(object):

    def __init__(self):
        self.dim = 21 #window size
        self.totaldim = self.dim*self.dim*3
        self.random = [] #stores the randomly chosen coordinates
        self.im1 = cv2.imread("textures/1.2.06.tiff")
        self.im2 = cv2.imread("textures/1.2.01.tiff")
        self.im3 = cv2.imread("textures/1.2.05.tiff")
        self.im4 = cv2.imread("textures/1.2.07.tiff")
        self.testim = cv2.imread("textures/combo3.tif")
        self.test = 3 #test number
        #dimensions of the test images
        self.test_x = 512 
        self.test_y = 512
        self.train_four()
    
    def train_two(self):
        """Training and testing on two images"""
        print("Started Training Two Image Classification")
        start = time.time()
        x_train = np.zeros((1, self.totaldim))
        y_train = []
        #trains based on original images
        for i in range(2000):
            if i % 2:
                im1train, upper_left = self.random_crop_and_reshape(self.im1, 512, 512)
                x_train = np.vstack((x_train, im1train))
                y_train += [1]
            else:
                im2train, upper_left = self.random_crop_and_reshape(self.im2, 512, 512)
                x_train = np.vstack((x_train, im2train))
                y_train += [2]
        #clf = svm.SVC(C = 100, kernel = "sigmoid", coef0 = 1.5)
        clf = svm.SVC(C=100., kernel ="poly", degree = 3)
        #clf = svm.LinearSVC(C= 100)
        print(clf)
        clf.fit(x_train[1:], y_train) 
        
        print("Finished training...")
        total = 4000
        self.random = [] #clear the list- resused in making a test
        self.make_test(total)
        self.y_pred = clf.predict(self.x_test)
        end = time.time()
        print("Time elapsed training and testing:", end - start)
        #Information on prediction 
        print("Accuracy score:", accuracy_score(self.y_test, self.y_pred))
        print(classification_report(self.y_test, self.y_pred))
        
        #makes a confusion matrix
        class_names = np.asarray(["1.2.06", "1.2.10"])
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, class_names, title='Confusion matrix, without normalization')
        plt.savefig("confusion matrix.png")
        plt.show()

        self.make_image()

    def train_four(self):
        """Training and testing on four images"""
        print("Started Training Multiclass Classification")
        start = time.time()
        x_train = np.zeros((1, self.totaldim))
        y_train = []
        x = 0
        for i in range(4000):
            if x == 4:
                x = 0
            if x == 3:
                im1train, upper_left = self.random_crop_and_reshape(self.im1, 512, 512)
                x_train = np.vstack((x_train, im1train))
                y_train += [1]
            elif x == 2:
                im2train, upper_left = self.random_crop_and_reshape(self.im2, 512, 512)

                x_train = np.vstack((x_train, im2train))
                y_train += [2]
            elif x == 1:
                im3train, upper_left = self.random_crop_and_reshape(self.im3, 512, 512)

                x_train = np.vstack((x_train, im3train))
                y_train += [3]
            else:
                im2train, upper_left = self.random_crop_and_reshape(self.im4, 512, 512)
                
                x_train = np.vstack((x_train, im2train))
                y_train += [4]
            x += 1
        clf = svm.SVC(C = 100, kernel ="poly", degree = 3)
        #clf = OneVsRestClassifier(estimator=SVC(random_state = 0, kernel = 'poly', degree = 3))
        print(clf)

        clf.fit(x_train[1:], y_train) 

        print("Finished training...")
        
        total = 4000
        self.make_test(total)
        self.y_pred = clf.predict(self.x_test)
        end = time.time()
        print("Time elapsed training and testing:", end - start)
        print("Accuracy:", accuracy_score(self.y_test, self.y_pred))
        print(classification_report(self.y_test, self.y_pred))

        class_names = np.asarray(["1.2.06", "1.2.01", "1.2.05", "1.2.07"])
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
    
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, class_names, title='Confusion matrix, without normalization')
        plt.savefig("confusion matrix.png")
        plt.show()
        self.make_image()
        
    def make_image(self):
        """Creates a visualization of the predictions"""
        graph = np.zeros((self.test_y,self.test_x,3), np.uint8)
        graph[:] = (255,255,255) #open cv uses bgr
        for coord in range(len(self.upper_left_array)):
            upper_left = self.upper_left_array[coord]
            lower_right = (upper_left[0] + self.dim, upper_left[1] + self.dim)
            if self.y_pred[coord] == [1]: #blue
                color = (120,0,0) 
            elif self.y_pred[coord] == [2]: #green
                color = (0,120,0)
            elif self.y_pred[coord] == [3]: #yellow
                color = (0,255,250)
            else:
                color = (0,120,255) #orange

            graph = cv2.rectangle(graph, upper_left, lower_right,color,-1)
        
        cv2.imshow('image',graph)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("graph2.png", graph)


    def make_test(self, total):
        """Makes a test vector with corresponding true values"""
        self.x_test = np.zeros((1,self.totaldim))
        self.y_test = []
        self.upper_left_array = []
        for i in range(total):
            im1test, upper_left = self.random_crop_and_reshape(self.testim, self.test_x, self.test_y)
            self.x_test = np.vstack((self.x_test, im1test))
            self.y_test_write(upper_left)
            self.upper_left_array.append(upper_left)
        self.x_test = self.x_test[1:]

    def y_test_write(self, coords):
        """Based off the test images it labels the coordinates"""
        x = coords[0]
        y = coords[1]
        if self.test == 1:
            if x < 512:
                self.y_test += [1]
            else:
                self.y_test += [2]

        elif self.test == 2:
            if x < 73:
                self.y_test +=[1]
            elif x < 512 and (y < 73 or y > 439):
                self.y_test +=[1]
            elif x > 146 and x < 219 and y > 146 and y < 366:
                self.y_test +=[1]
            elif x < 512 and x > 219 and ((y > 146 and y < 219) or (y > 293 and y < 366)):
                self.y_test +=[1]
            elif x > 878 and x < 951 and y > 73 and y < 439:
                self.y_test +=[1]
            elif x > 512 and x < 878 and ((y < 146 and y > 73) or (y > 366 and y < 439)):
                self.y_test +=[1]
            elif x > 512 and x < 805 and y > 219 and y < 293:
                self.y_test +=[1]
            else:
                self.y_test +=[2]
        
        elif self.test == 3:
            if y < 256:
                if x < 256:
                    self.y_test += [1]
                else:
                    self.y_test += [2]
            else:
                if x < 256:
                    self.y_test += [3]
                else:
                    self.y_test += [4]

    def random_crop_and_reshape(self, img, imgx, imgy):
        """Generates random coordinates to crop"""
        upper_x = random.randrange(imgx-self.dim)
        upper_y = random.randrange(imgy-self.dim)
        upper_left = (upper_x, upper_y)
        while upper_left in self.random:
            upper_x = random.randrange(imgx-self.dim)
            upper_y = random.randrange(imgy-self.dim)
            upper_left = (upper_x, upper_y)

        self.random.append(upper_left)
        newimg, upper_left = self.crop_and_reshape(img, upper_left)
        return newimg, upper_left

    def crop_and_reshape(self, img, upper_left):
        """Takes in an upper left corner and an images and crops the window size
            and reshapes it into a 1D array"""
        upper_y = upper_left[1]
        upper_x = upper_left[0]
        newimg = img[upper_y: upper_y + self.dim, upper_x: upper_x + self.dim]
        # Makes it a 1D array
        newimg = np.reshape(newimg, self.totaldim)
        return newimg, (upper_x, upper_y)

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Taken from http://scikit-learn.org/stable/auto_examples/model_selection/
        plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


if __name__ == '__main__':
    SVM()


