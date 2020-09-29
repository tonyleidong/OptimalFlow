from flask import Flask, render_template, flash, request, session,redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
import time
import os
import json

json_path = os.path.join(os.path.dirname("./"), 'webapp.json')
json_path_settings = os.path.join(os.path.dirname("./"), 'settings.json')
app  = Flask(__name__)

log_flag = False

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
    return render_template('index.html',thePercent = thePercent)

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/parameters/',methods=['POST','GET'])
def parameters():
    if request.method == "POST":
        try:
            confirm_reset = request.form['confirm_reset']
            with open(json_path_settings,'r',encoding='utf-8') as s_file:
                para = json.load(s_file)
            para['confirm_reset'] = confirm_reset
            s_file.close()
            w_file = open(json_path_settings, "w",encoding='utf-8')
            json.dump(para, w_file)
            w_file.close()
            
        except:
            try:
                algo_name = request.form['parent']
                para_name = request.form['child']
                para_val = request.form['paraValCls']
                para_val = list(para_val.split(","))
                try:
                    para_val = [float(i) if '.' in i else int(i) for i in para_val]
                except:
                    para_val = para_val
                with open(json_path_settings,'r',encoding='utf-8') as s_file:
                    para = json.load(s_file)
                para['space_set']['cls'][algo_name][para_name] = para_val
                para['confirm_reset'] = 'no_confirm'
                w_file = open(json_path_settings, "w",encoding='utf-8')
                json.dump(para, w_file)
                w_file.close()
                print(algo_name,para_name,para_val)
                s_file.close()
            except:
                try:
                    algo_name = request.form['parent2']
                    print(algo_name)
                    para_name = request.form['child2']
                    para_val = request.form['paraValReg']
                    para_val = list(para_val.split(","))
                    print(1)
                    try:
                        para_val = [float(i) if '.' in i else int(i) for i in para_val]
                    except:
                        para_val = para_val
                    print(para_val)
                    with open(json_path_settings,'r',encoding='utf-8') as s_file:
                        para = json.load(s_file)
                    para['space_set']['reg'][algo_name][para_name] = para_val
                    para['confirm_reset'] = 'no_confirm'
                    w_file = open(json_path_settings, "w",encoding='utf-8')
                    json.dump(para, w_file)
                    w_file.close()
                    print(algo_name,para_name,para_val)
                    s_file.close()
                except:
                    print("Error in Setting Searchin Space!")
                
    return render_template('parameters.html')

@app.route('/docs/')
def docs():
    return render_template('docs.html')

@app.route('/logs/',methods=['POST','GET'])
def logs():
    global log_flag
    if request.method == "POST":
        try:
            import time
            from win32com import client
            from os import listdir
            from os.path import isfile, join
            run_btn =request.form['run_btn']
            logfile =request.form['logfile']

            file_list = [f for f in listdir('./logs') if isfile(join('./logs', f))]
            file_select = [logfile]

            for item in file_list:
                ext_flag = [item.startswith(i) for i in file_select]
                
                if (ext_flag==[True]) and item.endswith('.log'):
                    contents = open("./logs/"+item,"r")
                    with open("./templates/logfile.html", "w") as e:
                        for lines in contents.readlines():
                            e.write("<pre>" + lines + "</pre> <br>\n")
            log_flag = True
        except:
                print('Read Log Files Error!')
        return render_template('logs.html',log_flag=log_flag)
    else:
        log_flag = False
        return render_template('logs.html',log_flag = log_flag)

@app.route('/nologfile/')
def nologfile():
    return render_template('nologfile.html')

@app.route('/logfile/')
def logfile():
    return render_template('logfile.html')

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




