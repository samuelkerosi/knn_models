import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#  testting and training files
training_file = np.genfromtxt("buyTraining.txt")
testing_file = np.genfromtxt("buyTesting.txt")

#  train stuff
X_train = training_file[:, :-1]
y_train = training_file[:, -1]
# test stuff
X_test = testing_file[:, :-1]
y_test = testing_file[:, -1]


# create classifier and train usng the training sets
knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_classifier.fit(X_train, y_train)

#  prediction using test data
predictiton = knn_classifier.predict(X_test)
print(predictiton)

# confuion matrices
cm = confusion_matrix(y_test, predictiton)
print("Confusion Matrix \n", cm)

# precision
precision = precision_score(y_test, predictiton)
print("precision: \n", precision)

# recall
recall = recall_score(y_test, predictiton,  average=None)
print("Reecall: \n", recall)

# accuracy
accuracy = metrics.accuracy_score(y_test, predictiton)
print("accuracy: \n", accuracy)

# output file
outfile = open("knnmodels_output.txt", "w")
outfile.write("Knn Model Output")
outfile.write("\n")
outfile.write("Precision: \n")
outfile.write(str(precision))
outfile.write("\n")
outfile.write("Recall: \n")
outfile.write(str(recall))
outfile.write("\n")
outfile.write("Confusion matrix: \n")
outfile.write(str(cm))
outfile.write("\n")
outfile.write("Accuracy: \n")
outfile.write(str(accuracy))
outfile.close()
