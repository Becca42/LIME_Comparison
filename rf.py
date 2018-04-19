"""
6700 Final Project

rebecca adara


NOTES & TODO for report:

 - why random forests?
 - using default for maxdepth, etc.

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lime
import lime.lime_tabular
import matplotlib
import matplotlib.pyplot as plt


names = ("id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst")

names1 = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

def loadData():
	"""
		loads data from csv into numpy array
		Returns:
			array tuple - data (all rows, all columns but id and diagnosis (label)) 
				& labels (diagnosis)
	"""
	filename = "data.csv"
	alldata = np.genfromtxt(filename, dtype=None, delimiter="," , skip_header=1, names=names1)
	dataArray =  np.asarray([alldata[name] for name in names1 if name != "id" and name != "diagnosis"])
	# labelArray = [row[1] for row in alldata]
	labelArray = alldata['diagnosis']

	# TODO split off test set
	index = int(len(dataArray[0]) * 0.8)
	# X = dataArray[:index].reshape((-1, len(names1)-2))
	dataTrain = alldata[:index]
	dataTest = alldata[index+1:]
	X = np.asarray([dataTrain[name] for name in names1 if name != "id" and name != "diagnosis"]).transpose()
	y = labelArray[:index]
	Xtest =  np.asarray([dataTest[name] for name in names1 if name != "id" and name != "diagnosis"]).transpose()
	ytest = labelArray[index + 1:]

	print(len(X[0]))
	print(X[0][0])

	return X, y, Xtest, ytest

def fitClassifier(X, y):
	"""
		fits classifier to data and lables
		Parameters:
			X - n x m array - data
			y - n x 1 array - labels
		Returns:
			fitted classifier
	"""
	clf = RandomForestClassifier()
	clf.fit(X,y)
	return clf

def checkAccuracy(Xtest, ytest, classifier):
	"""
		returns accruacy of classifier on test set
		Parameters:
			Xtest - m x n numpy array - test values
			ytest - n x 1 numpy array - test lables
			classifier - trained classifier

		Returns:
			float - accuracy 
	"""
	assert(len(ytest) == Xtest.shape[0])
	totalSamples = len(ytest)
	correct = 0.
	for i in range(0, totalSamples):
		classification = classifier.predict([Xtest[i]])
		if classification[0] == ytest[i]:
			correct += 1.
	return correct/totalSamples

def main():
	X, y, Xtest, ytest = loadData()
	classifier = fitClassifier(X, y)
	print("Done Training")
	# check accuracy
	accuracy = checkAccuracy(Xtest, ytest, classifier)
	print("Accuracy: " + str(accuracy))
	# TODO explain classifier
	# explain(X, Xtest, classifier)

if __name__ == "__main__":
    main()