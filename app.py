from flask import Flask, request, render_template, send_file
import pandas as pd
from io import BytesIO
import xlsxwriter
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
  return render_template('index.html')

# Route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
  # Get the form data and file
  limit_bal = request.form['limit_bal']
  sex = request.form['sex']
  education = request.form['education']
  marriage = request.form['marriage']
  age = request.form['age']
  pay_1 = request.form['pay_1']
  pay_2 = request.form['pay_2']
  pay_3 = request.form['pay_3']
  pay_4 = request.form['pay_4']
  pay_5 = request.form['pay_5']
  pay_6 = request.form['pay_6']
  bill_amt_1 = request.form['bill_amt_1']
  bill_amt_2 = request.form['bill_amt_2']
  bill_amt_3 = request.form['bill_amt_3']
  bill_amt_4 = request.form['bill_amt_4']
  bill_amt_5 = request.form['bill_amt_5']
  bill_amt_6 = request.form['bill_amt_6']
  pay_amt_1 = request.form['pay_amt_1']
  pay_amt_2 = request.form['pay_amt_2']
  pay_amt_3 = request.form['pay_amt_3']
  pay_amt_4 = request.form['pay_amt_4']
  pay_amt_5 = request.form['pay_amt_5']
  pay_amt_6 = request.form['pay_amt_6']
  file = request.files

# Check if a file was uploaded
if file:
  # Read the file into a pandas dataframe
  df = pd.read_excel(file)
else:
  # Create a dataframe from the form data
  data = {
    'Limit Balance': [limit_bal],
    'Sex': [sex],
    'Education': [education],
    'Marriage': [marriage],
    'Age': [age],
    'Pay 1': [pay_1],
    'Pay 2': [pay_2],
    'Pay 3': [pay_3],
    'Pay 4': [pay_4],
    'Pay 5': [pay_5],
    'Pay 6': [pay_6],
    'Bill Amount 1': [bill_amt_1],
    'Bill Amount 2': [bill_amt_2],
    'Bill Amount 3': [bill_amt_3],
    'Bill Amount 4': [bill_amt_4],
    'Bill Amount 5': [bill_amt_5],
    'Bill Amount 6': [bill_amt_6],
    'Pay Amount 1': [pay_amt_1],
    'Pay Amount 2': [pay_amt_2],
    'Pay Amount 3': [pay_amt_3],
    'Pay Amount 4': [pay_amt_4],
    'Pay Amount 5': [pay_amt_5],
    'Pay Amount 6': [pay_amt_6]
  }
  df = pd.DataFrame(data)

# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Construct new features using PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

# Extract the most important features using PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

# Initialize the LogisticRegression and SVC models
lr = LogisticRegression()
svc = SVC()

# Fit the LogisticRegression and SVC models on the transformed data
lr.fit(X_pca, y)
svc.fit(X_pca, y)

# Make predictions using the models
y_pred_lr = lr.predict(X_pca)
y_pred_svc = svc.predict(X_pca)

# Create a pandas dataframe with the predictions
predictions = pd.DataFrame({
  'Logistic Regression': y_pred_lr,
  'SVC': y_pred_svc
})

# Create a buffer to write the data to a memory object
buffer = BytesIO()

# Create the Excel writer and write the data to the buffer
writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
predictions.to_excel(writer, index=False)

# Save the Excel file and close the writer
writer.save()
buffer.seek(0)

# Send the file to the client
return send_file(
  buffer,
  attachment_filename='predictions.xlsx',
  as_attachment=True
)

# Run the app
if __name__ == '__main__':
  app.run()
