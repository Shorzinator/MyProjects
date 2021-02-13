# Naive Bayes

def NaiBay(X_train, y_train, X_test, y_test):

	# Building the model
	from sklearn.naive_bayes import GaussianNB
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)

	# Predicting the Test set results for Naive Bayes
	y_pred = classifier.predict(X_test)
	print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	# Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix, accuracy_score
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	accuracy_score(y_test, y_pred)