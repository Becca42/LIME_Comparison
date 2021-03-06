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
		# store result, actual in features
		features.append({"actual": actual, "prediction": prediction[0], "exp": results})

	print("")
	print(x)
	print("actuat: " + str(actual))
	print("prediction" + str(prediction))
	exp.as_pyplot_figure().show() # NOTE defaults to label=1
	graph = exp.as_pyplot_figure()
	print("graph")
	print(graph)
	graph.show()
	#plot_features(exp, ncol = 1)

	X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
	C,S = np.cos(X), np.sin(X)

	plt.plot(X,C)
	plt.plot(X,S)

	plt.show()
	return features

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
	print correct
	print totalSamples
	return correct/totalSamples


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
	featureUseCount = {} # aggregate confidences for all features for all predictions

	for f in features:
		# check if correct prediction was made
		label = f['actual']
		if label == f['prediction']:
			correctByPrediction[label].append(f['exp'])
			for (feat, confidence) in f['exp']:
				confidenceList = correctByPredictionByFeature[label].get(feat, [])
				confidenceList.append(confidence)
				correctByPredictionByFeature[label][feat] = confidenceList
				# add to totally feature counts
				confidenceListAll = featureUseCount.get(feat, [])
				confidenceListAll.append(confidence)
				featureUseCount[feat] = confidenceListAll
		else:
			incorrectByPrediction[f['actual']].append(f['exp'])
			for (feat, confidence) in f['exp']:
				confidenceList = incorrectByPredictionByFeature[label].get(feat, [])
				confidenceList.append(confidence)
				incorrectByPredictionByFeature[label][feat] = confidenceList
				# add to totall feature counts
				confidenceListAll = featureUseCount.get(feat, [])
				confidenceListAll.append(confidence)
				featureUseCount[feat] = confidenceListAll
	useCounts = {}
	for k in featureUseCount.keys():
		useCounts[k] = len(featureUseCount[k])

	#print('useCounts')
	#print(useCounts)

	# get most common features used across all predictions (correct and incorrect)
	sortedByMostUsed = sorted(featureUseCount, key=lambda k: len(featureUseCount[k]), reverse=True)
	#print("sortedByMostUsed")
	#print(sortedByMostUsed)
	# TODO get most common features used for each class (M/B) (correct and incorrect)
	# get features sorted by avg importance (weight of explainer)
	#sortedByAvgImportance = sorted(featureUseCount, key=lambda k: sum(featureUseCount[k])/len(featureUseCount[k]), reverse=True)
	#print(sortedByAvgImportance)

	#print('Average Importance')
	avgimportance = {}
	for k in featureUseCount.keys():
		avgimportance[k] = sum(featureUseCount[k])/len(featureUseCount[k])
	#print(avgimportance)

def main():
	X, y, Xtest, ytest = loadData()
	classifier = fitClassifier(X, y)
	print("Done Training")
	# check accuracy
	accuracy = checkAccuracy(Xtest, ytest, classifier)
	print("Accuracy: " + str(accuracy))
	# TODO explain classifier
	features = explain(X, Xtest, ytest, classifier)
	lookAtFeatures(features)

if __name__ == "__main__":
    main()