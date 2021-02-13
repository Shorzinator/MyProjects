# Kernel Support Vector Machine

def KSVM(X_train, y_train, X_test, y_test):
	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	# Training the model on the 'rbf' kernel
	from sklearn.svm import SVC
	classifier = SVC(kernel = 'rbf', random_state = 0)
	classifier.fit(X_train, y_train)

	# Predicting the Test set results for SVM
	y_pred = classifier.predict(X_test)
	print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

	# Making confusion matrix
	from sklearn.metrics import confusion_matrix, accuracy_score
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	accuracy_score(y_test, y_pred)