{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of training dataset: (1797, 64)\n",
      "Shape of testing dataset: (1797,)\n"
     ]
    }
   ],
   "source": [
    "#importing dataset\n",
    "from sklearn.datasets import load_digits\n",
    "digit = load_digits()\n",
    "print(\"Shape of training dataset:\",digit.data.shape) #1797 64*1--array, 8*8-- images\n",
    "print(\"Shape of testing dataset:\",digit.target.shape)\n",
    "# In order to utilize an 8x8 figure like this, we’d have to first transform it into a feature vector with length 64."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ".. _digits_dataset:\n",
      "\n",
      "Optical recognition of handwritten digits dataset\n",
      "--------------------------------------------------\n",
      "\n",
      "**Data Set Characteristics:**\n",
      "\n",
      "    :Number of Instances: 5620\n",
      "    :Number of Attributes: 64\n",
      "    :Attribute Information: 8x8 image of integer pixels in the range 0..16.\n",
      "    :Missing Attribute Values: None\n",
      "    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)\n",
      "    :Date: July; 1998\n",
      "\n",
      "This is a copy of the test set of the UCI ML hand-written digits datasets\n",
      "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits\n",
      "\n",
      "The data set contains images of hand-written digits: 10 classes where\n",
      "each class refers to a digit.\n",
      "\n",
      "Preprocessing programs made available by NIST were used to extract\n",
      "normalized bitmaps of handwritten digits from a preprinted form. From a\n",
      "total of 43 people, 30 contributed to the training set and different 13\n",
      "to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of\n",
      "4x4 and the number of on pixels are counted in each block. This generates\n",
      "an input matrix of 8x8 where each element is an integer in the range\n",
      "0..16. This reduces dimensionality and gives invariance to small\n",
      "distortions.\n",
      "\n",
      "For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.\n",
      "T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.\n",
      "L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,\n",
      "1994.\n",
      "\n",
      ".. topic:: References\n",
      "\n",
      "  - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their\n",
      "    Applications to Handwritten Digit Recognition, MSc Thesis, Institute of\n",
      "    Graduate Studies in Science and Engineering, Bogazici University.\n",
      "  - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.\n",
      "  - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.\n",
      "    Linear dimensionalityreduction using relevance weighted LDA. School of\n",
      "    Electrical and Electronic Engineering Nanyang Technological University.\n",
      "    2005.\n",
      "  - Claudio Gentile. A New Approximate Maximal Margin Classification\n",
      "    Algorithm. NIPS. 2000.\n"
     ]
    }
   ],
   "source": [
    "print(digit.DESCR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digit.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = digit.data # independent variables matrix\n",
    "y = digit.target      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1797, 64)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.ndim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plt.figure(figsize=(2,2))\n",
    "plt.imshow(X[0].reshape(8,8),cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,\n",
    "                                                random_state=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1437, 64)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.,  0.,  8., 14., 16., 16.,  1.,  0.,  0.,  6., 16., 16.,  8.,\n",
       "        3.,  0.,  0.,  0., 14., 14.,  1.,  0.,  0.,  0.,  0.,  0., 10.,\n",
       "       15.,  4.,  0.,  0.,  0.,  0.,  0.,  3., 15., 16.,  6.,  0.,  0.,\n",
       "        0.,  0.,  0.,  1.,  8., 15.,  2.,  0.,  0.,  0.,  0.,  2., 13.,\n",
       "       15.,  0.,  0.,  0.,  0.,  0., 10., 16.,  4.,  0.,  0.,  0.])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test = X_test[0]\n",
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1529e875888>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X_train[:,0],X_train[:,1],c=y_train)\n",
    "plt.scatter(test[0],test[1],color='black')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import KFold,cross_val_score\n",
    "kf = KFold(10,shuffle=True,random_state=10)\n",
    "          # k = no. of subsets = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Applying SVM model to training dataset\n",
    "from sklearn.svm import SVC\n",
    "classifier=SVC(kernel='rbf',gamma=0.001) #hyperparameter tuning c and gamma values\n",
    "#linear kernel only gives 91.8% accuracy\n",
    "#polynomial kernel with degree=3 gives 86% accuracy\n",
    "#sigmoid kernel gives 91.1% accuracy\n",
    "#rbf kernel gives 97.816% accuracy so i used rbf kernel\n",
    "model = classifier.fit(X_train,y_train)\n",
    "# model = classifier.fit(X_dataset_train,train_lbl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted output values: [5 2 5 4 8 2 4 3 3 0 8 7 0 1 8 6 9 7 9 7 1 8 6 7 8 8 5 3 5 9 3 3 7 3 4 1 9\n",
      " 2 5 4 2 1 0 9 2 3 6 1 9 4 4 9 8 4 8 5 9 7 1 0 4 5 8 4 7 9 0 7 1 3 9 3 3 8\n",
      " 0 7 3 6 5 2 0 8 8 0 1 1 2 8 8 8 2 6 3 4 7 9 8 2 9 2 5 0 8 0 4 8 8 0 6 7 3\n",
      " 3 9 1 5 4 6 0 8 8 1 1 7 9 9 5 2 3 3 8 7 6 2 5 4 3 3 7 6 7 2 7 4 9 5 1 9 4\n",
      " 6 1 1 1 4 0 8 9 1 2 3 5 0 3 4 1 5 4 9 3 5 6 4 0 8 6 7 0 9 9 4 7 3 5 2 0 6\n",
      " 7 5 3 9 7 1 3 2 8 3 3 1 7 1 1 1 7 1 6 7 6 9 5 2 3 5 2 9 5 4 8 2 9 1 5 0 2\n",
      " 3 9 0 2 0 2 1 0 5 0 6 4 2 1 9 0 9 0 6 9 4 4 9 7 5 6 1 8 7 0 8 6 2 0 1 2 3\n",
      " 8 4 4 3 5 7 9 7 2 0 2 0 9 2 8 6 3 6 0 6 6 6 7 1 6 1 7 6 0 6 3 7 4 6 2 8 0\n",
      " 8 4 7 3 3 0 0 2 3 9 7 4 6 7 9 7 6 0 5 6 2 7 1 0 5 1 6 4 7 2 5 1 4 6 6 5 0\n",
      " 2 9 8 7 9 6 7 0 8 3 5 9 4 1 5 5 4 7 3 9 2 7 3 3 6 6 3]\n"
     ]
    }
   ],
   "source": [
    "#Predicting results of training dataset\n",
    "y_pred=classifier.predict(X_test)\n",
    "print(\"Predicted output values:\",y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 2, 5, 4, 8, 2, 4, 3, 3, 0, 8, 7, 0, 1, 8, 6, 9, 7, 9, 7, 1, 8,\n",
       "       6, 7, 8, 8, 5, 3, 5, 9, 3, 3, 7, 3, 4, 1, 9, 2, 5, 4, 2, 1, 0, 9,\n",
       "       2, 3, 6, 1, 9, 4, 4, 9, 8, 4, 8, 5, 9, 7, 1, 0, 4, 5, 8, 4, 7, 9,\n",
       "       0, 7, 1, 3, 9, 3, 3, 8, 0, 7, 3, 6, 5, 2, 0, 8, 8, 0, 1, 1, 2, 8,\n",
       "       8, 8, 2, 6, 3, 4, 7, 9, 8, 2, 9, 2, 5, 0, 8, 0, 4, 8, 8, 0, 6, 7,\n",
       "       3, 3, 9, 1, 5, 4, 6, 0, 8, 8, 1, 1, 7, 9, 9, 5, 2, 3, 3, 8, 7, 6,\n",
       "       2, 5, 4, 3, 3, 7, 6, 7, 2, 7, 4, 9, 5, 1, 9, 4, 6, 1, 1, 1, 4, 0,\n",
       "       8, 9, 1, 2, 3, 5, 0, 3, 4, 1, 5, 4, 9, 3, 5, 6, 4, 0, 8, 6, 7, 0,\n",
       "       9, 9, 4, 7, 3, 5, 2, 0, 6, 7, 5, 3, 9, 7, 1, 3, 2, 8, 3, 3, 1, 7,\n",
       "       1, 1, 1, 7, 1, 6, 7, 6, 9, 5, 2, 3, 5, 2, 9, 5, 4, 8, 2, 9, 1, 5,\n",
       "       0, 2, 3, 9, 0, 2, 0, 2, 1, 0, 5, 0, 6, 4, 2, 1, 9, 0, 9, 0, 6, 9,\n",
       "       4, 4, 9, 7, 5, 6, 1, 8, 7, 0, 8, 6, 2, 0, 1, 2, 3, 8, 4, 4, 3, 5,\n",
       "       7, 9, 7, 2, 0, 2, 0, 9, 2, 8, 6, 3, 6, 0, 6, 6, 6, 7, 1, 6, 1, 7,\n",
       "       6, 0, 6, 3, 7, 4, 6, 2, 8, 0, 8, 4, 7, 3, 3, 0, 0, 2, 3, 9, 7, 4,\n",
       "       6, 7, 9, 7, 6, 0, 5, 6, 2, 7, 1, 0, 5, 1, 6, 4, 7, 2, 5, 1, 4, 6,\n",
       "       6, 5, 0, 2, 9, 8, 7, 9, 6, 7, 0, 8, 3, 5, 9, 4, 1, 5, 5, 4, 7, 3,\n",
       "       9, 2, 7, 3, 3, 6, 6, 3])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred.flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([ 58, 129, 154], dtype=int64),)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where(y_test!=y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(360,)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 99.16666666666667 \n",
      "\n",
      "[[37  0  0  0  0  0  0  0  0  0]\n",
      " [ 0 34  0  0  0  0  0  0  0  0]\n",
      " [ 0  0 34  0  0  0  0  0  0  0]\n",
      " [ 0  0  0 40  0  0  0  0  0  0]\n",
      " [ 0  0  0  0 33  0  0  0  1  0]\n",
      " [ 0  0  0  0  0 32  0  0  0  0]\n",
      " [ 0  0  0  0  0  0 37  0  0  0]\n",
      " [ 0  0  0  0  0  0  0 40  0  0]\n",
      " [ 0  1  0  0  0  0  0  0 32  0]\n",
      " [ 0  0  0  0  0  0  0  0  1 38]]\n"
     ]
    }
   ],
   "source": [
    "# confusion matrix and accuracy\n",
    "\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import confusion_matrix\n",
    "# accuracy\n",
    "print(\"Accuracy:\", metrics.accuracy_score(y_true=y_test, y_pred=y_pred)*100, \"\\n\")\n",
    "\n",
    "# cm\n",
    "print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Predicted')"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting some samples as well as converting into matrix\n",
    "y_pred = y_pred.reshape(12,30)\n",
    "plt.imshow(y_pred, cmap='gray') # to remove color channel from input image\n",
    "plt.title(\"Predicted\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['model_digits.pkl']"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "joblib.dump(model, \"model_digits.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
