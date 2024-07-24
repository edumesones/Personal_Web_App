from flask import Flask, render_template,send_file,abort,url_for,request
import os
import pandas as pd
import subprocess
import nbconvert
from nbconvert import PDFExporter
import pdfkit
import nbformat
import tempfile

from sklearn.preprocessing import LabelEncoder
import os
from model_utils import read_data, preprocess_data, train_model
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cv')
def cv():
    return render_template('cv.html')

@app.route('/python-projects')
def python_projects():
    return render_template('python_projects.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



data = read_data(os.path.join(os.path.dirname(__file__), 'data_to_practice', 'data_deposit.csv'))
X, y, label_encoders = preprocess_data(data)
print(label_encoders)
model = train_model(X, y)



@app.route('/ml_model', methods=['GET', 'POST'])
def ml_model():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict()
        print(form_data)
        input_data = pd.DataFrame([form_data],columns=X.columns)

        # Preprocess input data
        for column in input_data.columns:
            if column in label_encoders:
                # Handle new categories
                input_data[column] = input_data[column].map(lambda x: x if x in label_encoders[column].classes_ else 'Unknown')
                
                # Transform using LabelEncoder
                input_data[column] = label_encoders[column].transform(input_data[column])
            elif input_data[column].dtype == 'object':
                # Convert to numeric if possible
                input_data[column] = pd.to_numeric(input_data[column], errors='ignore')

        # Ensure all columns from training data are present
        for column in X.columns:
            if column not in input_data.columns:
                input_data[column] = 0

        # Reorder columns to match training data
        print(input_data)
        input_data = input_data.reindex(columns=X.columns)

        # Predict
        try:
            prediction = model.predict_proba(input_data)[:, 1][0]
            print('pasa')
            print(prediction)
        except ValueError as e:
            
            return render_template('ml_model.html', error=str(e), form_data=form_data)
        return render_template('ml_model.html', prediction=prediction, form_data=form_data)

    return render_template('ml_model.html', prediction=None, form_data=None)

@app.route('/working-now')
def working_now():
    return render_template('working_now.html')

@app.route('/convert/<path:filename>')
def convert_to_pdf(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    notebook_path = os.path.join(base_dir, 'notebooks', filename)
    html_exporter = nbconvert.HTMLExporter()
    html_content, _ = html_exporter.from_filename(notebook_path)

    # Create a temporary file to save the PDF
    temp_html_path = os.path.join(os.path.dirname(__file__), f"{filename.rsplit('.', 1)[0]}.html")
    with open(temp_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Serve the HTML file
    return send_file(temp_html_path)

@app.route('/download/<path:filename>')
def download_csv(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.normpath(os.path.join(base_dir, 'data_to_practice', filename))
    if not os.path.exists(file_path):
        return abort(404, description="File not found")
    return send_file(file_path, as_attachment=True)

@app.route('/view_imports')
def view_imports():
    scripts_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'py_scripts')
    py_files = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
    file_links = [url_for('view_py_file', filename=f) for f in py_files]
    return render_template('view_imports.html', file_links=file_links, py_files=py_files)

@app.route('/view_py/<path:filename>')
def view_py_file(filename):
    scripts_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'py_scripts')
    file_path = os.path.join(scripts_dir, filename)
    if not os.path.exists(file_path):
        return abort(404, description="File not found")
    with open(file_path, 'r') as file:
        content = file.read()
    return render_template('view_file.html', filename=filename, content=content)
if __name__ == '__main__':
    app.run(debug=True)