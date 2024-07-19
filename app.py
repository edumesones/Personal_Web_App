from flask import Flask, render_template,send_file,abort,url_for
import os
import subprocess
import nbconvert
from nbconvert import PDFExporter
import pdfkit
import nbformat
import tempfile
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

@app.route('/ml-model')
def ml_model():
    return render_template('ml_model.html')

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