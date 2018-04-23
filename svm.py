"""
6700 Final Project

rebecca adara


NOTES & TODO for report:

 - split DS into train/test/val (will want cross-val for svm with this small of DS - TODO find citation for why)
 - SVM good for smaller DS (TODO find citation for why)
 - TODO explain why chose linear kernel and why chose C = 1.0 in fitClassifier
 - Accuracy: 0.946902654867

"""

import numpy as np
from sklearn import svm
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
	clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
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


def explain(X, Xtest, yTest, classifier):
	"""
		TODO explain features used to make predictions for all test points
		TODO keep track of where the prediction is correct
	"""

	features = [] # TODO fill with predictions used for each test-point, it's predicted label, and its actual label
	labelOrder = {'B': 0, 'M': 1}

	explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=names1[2:], class_names=["B", "M"]) # TODO B, M or M, B?? (seems to matter)
	
	# print(exp.available_labels())

	for i in range(0, len(Xtest)-1):
		x = Xtest[i]
		actual = yTest[i]
		prediction = classifier.predict([x])
		exp = explainer.explain_instance(x, classifier.predict_proba, labels=[0, 1])
		results = exp.as_list(label=labelOrder[prediction[0]])
		# TODO store result, actual in features
		features.append({"actual": actual, "prediction": prediction[0], "exp": results})

	return features
	# exp.as_pyplot_figure().show() # NOTE defaults to label=1
	# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
	# C,S = np.cos(X), np.sin(X)

	# plt.plot(X,C)
	# plt.plot(X,S)

	# plt.show()


def lookAtFeatures(features):
	"""
		TODO look at features used to make predictions, draw some conclusions
		Parameters:
			features - list of dictionaries - predictions and associated explanations
		Returns:
			TODO
	"""
	# collect features used to explain correct & incorrect predictions by label
	correctByPrediction = {'M': [], 'B': []}
	correctByPredictionByFeature = {'M': {}, 'B': {}}
	incorrectByPrediction = {'M': [], 'B': []}
	incorrectByPredictionByFeature = {'M': {}, 'B': {}}

	for f in features:
		label = f['actual']
		if label == f['prediction']:
			correctByPrediction[label].append(f['exp'])
			for (feat, exp) in f['exp']:
				expList = correctByPredictionByFeature[label].get(feat, [])
				correctByPredictionByFeature[label] = expList.append(exp)
		else:
			incorrectByPrediction[f['actual']].append(f['exp'])
			for (feat, exp) in f['exp']:
				expList = incorrectByPredictionByFeature[label].get(feat, [])
				incorrectByPredictionByFeature[label] = expList.append(exp)


def main():
	X, y, Xtest, ytest = loadData()
	classifier = fitClassifier(X, y)
	print("Done Training")
	# check accuracy
	accuracy = checkAccuracy(Xtest, ytest, classifier)
	print("Accuracy: " + str(accuracy))
	# TODO explain classifier
	features = explain(X, Xtest, ytest, classifier)
	# TODO do something with features; draw conclusions about features used to make predictions are the accuracy of those predictions
	print(features[0])

if __name__ == "__main__":
    main()