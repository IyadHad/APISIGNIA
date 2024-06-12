from flask import Flask, render_template, request, url_for, redirect
import requests

app = Flask(__name__)

API_URL = 'https://apisignia-n7fabltiz-iyadhads-projects.vercel.app/prediction'  # URL de votre API Flask

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Envoyer le fichier à l'API Flask
        try:
            files = {'file': file}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                #signe_predit = request.args.get('signe', 'No sign detected')
                return redirect(url_for('res', signe_predit=result))
            else:
                return redirect(url_for('res', error=f'Error: {response.status_code}',signe_predit=f'Error: {response.status_code}'))
        except Exception as e:
            return render_template('index.html', error=f'Error: {str(e)}')
    return render_template('index.html')

@app.route('/<signe_predit>',methods = ['post'])
def res(signe_predit):
    error = request.args.get('error', None)
    return render_template('result.html', signe_predit=signe_predit, error=error)   

if __name__ == '__main__':
    app.run(debug=True)
