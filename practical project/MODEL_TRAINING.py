import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
data = pd.read_csv('hearing_loss_full_data2set.csv')

X = data[['250Hz','500Hz','1000Hz','2000Hz','3000Hz','4000Hz','6000Hz','8000Hz']]
y = data['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model2 = SVC(kernel='linear', C=0.1)
model2.fit(X_train, y_train)
y2_pred = model2.predict(X_test)
print(confusion_matrix(y_test, y2_pred))
print(classification_report(y_test, y2_pred))
print(accuracy_score(y_test, y2_pred))
model = LogisticRegression(multi_class='ovr', max_iter=10000)
ds = scaler.transform([[1,1,2,1,1,1,1,9]])

model.fit(X_train, y_train)
ys=model.predict(ds)
print(ys)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy*100}%")
print("Classification Report:\n", report)