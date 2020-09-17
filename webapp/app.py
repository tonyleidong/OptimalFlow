from flask import Flask, render_template, flash, request, session,redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
import time
import os
# from os import listdir
# from os.path import isfile, join
# import win32com.client as comclt
import json

json_path = os.path.join(os.path.dirname("./"), 'webapp.json')
app  = Flask(__name__)
#creating routes
@app.route('/',methods=['POST', 'GET'])
def index():
    thePercent = 0
    if request.method == "POST":
        try:

            f_name = request.form['filename']
            label_col = request.form['label_col']
            with open(json_path,'r',encoding='utf-8') as d_file:
                para = json.load(d_file)
            para['filename'] = f_name
            para['label_col'] = label_col
            w_file = open(json_path, "w",encoding='utf-8')
            json.dump(para, w_file)
            w_file.close()

            print('Load Dataset Done:')
            print(f_name,label_col)
            thePercent = 25
        
        except:

            try:
                feature_num = request.form['feature_num']
                model_type_fs = request.form['model_type_fs']
                algo_fs = request.form.getlist('algo_fs')

                with open(json_path,'r',encoding='utf-8') as d_file:
                    para = json.load(d_file)
                para['autoFS']['feature_num'] = feature_num
                para['autoFS']['model_type_fs'] = model_type_fs
                para['autoFS']['algo_fs'] = algo_fs
                w_file = open(json_path, "w",encoding='utf-8')
                json.dump(para, w_file)
                w_file.close()
                print('autoFS Settings Done:')
                print(feature_num,model_type_fs,algo_fs)
                thePercent = 75
            except:

                try:
                    encode_band = request.form['encode_band']
                    model_type_pp = request.form['model_type_pp']
                    winsorizer = request.form.getlist('winsorizer')
                    sparsity = request.form['sparsity']
                    cols = request.form['cols']
                    scaler = request.form.getlist('scaler')
                    low_encode = request.form.getlist('low_encode')
                    high_encode = request.form.getlist('high_encode')
                    with open(json_path,'r',encoding='utf-8') as d_file:
                        para = json.load(d_file)
                    para['autoPP']['encode_band'] = encode_band
                    para['autoPP']['model_type_pp'] = model_type_pp
                    para['autoPP']['winsorizer'] = winsorizer
                    para['autoPP']['sparsity'] = sparsity
                    para['autoPP']['cols'] = cols
                    para['autoPP']['scaler'] = scaler
                    para['autoPP']['low_encode'] = low_encode
                    para['autoPP']['high_encode'] = high_encode
                    w_file = open(json_path, "w",encoding='utf-8')
                    json.dump(para, w_file)
                    w_file.close()
                    print('autoPP Settings Done:')
                    print(encode_band,model_type_pp,scaler,winsorizer,sparsity,cols,low_encode,high_encode)
                    
                    thePercent = 50
                except:
                    try:
                        model_type_cv = request.form['model_type_cv']
                        method_cv = request.form['method_cv']
                        algo_cv = request.form.getlist('algo_cv')
                        with open(json_path,'r',encoding='utf-8') as d_file:
                            para = json.load(d_file)
                        para['autoCV']['model_type_cv'] = model_type_cv
                        para['autoCV']['method_cv'] = method_cv
                        para['autoCV']['algo_cv'] = algo_cv
                        w_file = open(json_path, "w",encoding='utf-8')
                        json.dump(para, w_file)
                        w_file.close()
                        print('autoCV Settings Done:')
                        print(model_type_cv,method_cv,algo_cv)
                        thePercent = 100
                    except:
                        try:
                            run_btn =request.form['run_btn']
                            import time
                            import win32com.client as comclt
                            shell = comclt.Dispatch("WScript.Shell")
                            shell.run("cmd.exe")
                            time.sleep(1)
                            shell.SendKeys("webapp_script.py {ENTER}")
                        except:
                            print('Parameters Setting Error!')
    
    return render_template('index.html',thePercent = thePercent)

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/docs/')
def docs():
    return render_template('docs.html')

@app.route('/logs/',methods=['POST','GET'])
def logs():
    if request.method == "POST":
        try:
            run_btn =request.form['run_btn']
            logfile =request.form['logfile']
            import time
            from win32com import client
            from os import listdir
            from os.path import isfile, join
            file_list = [f for f in listdir('./logs') if isfile(join('./logs', f))]
            file_select = [logfile]
            
            for item in file_list:
                ext_flag = [item.startswith(i) for i in file_select]
                
                if (ext_flag==[True]) and item.endswith('.log'):
                    shell1 = client.Dispatch("WScript.Shell")
                    shell1.run("cmd.exe")
                    time.sleep(1)
                    shell1.SendKeys("cd ./logs {ENTER}")
                    shell1.SendKeys(item+" {ENTER}")
                    shell1.SendKeys("exit {ENTER}")
        except:
                print('Read Log Files Error!')
        return render_template('logs.html')
    return render_template('logs.html')
@app.route('/viz/')
def viz():
    return render_template('viz.html')

@app.route('/report/')
def report():
    return render_template('report.html')

@app.route('/diagram/')
def diagram():
    return render_template('diagram.html')

#run flask app
if __name__ == "__main__":
    app.run(debug=True)



# LOG_FOLDER = './logs'
# app.secret_key='secret key'
# app.config['LOG_FOLDER'] = './logs'
# ALLOWED_EXTENSIONS = set(['log'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# c_flag = False
# logfile = 'autoCV_log_sample.log'

# @app.route('/extract/')
# def logs():
#     global c_flag
#     extracted_values={}
#     if c_flag:
#         extracted_values = {'yes':1}
#         c_flag = False
#     return render_template('logs.html',logfile = logfile, extracted_values=extracted_values)


# @app.route('/extract',methods=['POST'])
# def select_log():
#     global c_flag
#     session.pop('_flashes',None)

#     if request.method == 'POST':
        
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
        
#         file = request.files['file']

#         if file.filename == '':
#             print('no')
#             flash('No File Selected For Review')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['LOG_FOLDER'],filename))
#             flash(f'{filename}')
#             global c_flag 
#             c_flag = True
#             global logfile
#             logfile = filename            
#             return redirect('/extract/')
#         else:
#             flash('Allowed file type is only *.log')
#             return(request.url)



