
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from prediction import detect_deepfake

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recentcases')
def recentcases():
    return render_template('recentcases.html')

@app.route('/impactofdeepfakes')
def impactofdeepfakes():
    return render_template('impactofdeepfakes.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        analysis_result = detect_deepfake(filepath)

        if 'error' in analysis_result:
            flash('Error in processing the video.')
            return redirect(url_for('index'))

        return render_template('result.html', 
                               result=analysis_result['result'],
                               confidence_real=analysis_result['confidence_real'],
                               confidence_deepfake=analysis_result['confidence_deepfake'])


if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# import os
# from model import predict_deepfake  # Import the prediction function from model.py

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.secret_key = 'supersecretkey'

# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recentcases')
# def recentcases():
#     return render_template('recentcases.html')

# @app.route('/impactofdeepfakes')
# def impactofdeepfakes():
#     return render_template('impactofdeepfakes.html')

# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'video' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
    
#     file = request.files['video']
    
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
    
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Call the deepfake detection function from model.py
#         analysis_result = predict_deepfake(filepath)

#         if 'error' in analysis_result:
#             flash('Error in processing the video.')
#             return redirect(url_for('index'))

#         return render_template('result.html', 
#                                result=analysis_result['verdict'],
#                                confidence_real=analysis_result['real_probability'],
#                                confidence_deepfake=analysis_result['deepfake_probability'])


# if __name__ == '__main__':
#     app.run(debug=True)
