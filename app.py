  
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
MW_Model = joblib.load('MW_model_initial.pk1')

##constants for scaling data
means = np.array([7.43397211e-01, 3.76955848e+02, 4.80482260e+01, 4.69922617e-01,
        1.34120357e+00, 2.84791054e+00, 1.18276824e+00, 3.98294118e+00,
        2.81976599e+00, 4.11944025e+00, 7.58886332e+00, 7.56022984e+01,
        4.06212388e-02, 4.26465459e-03, 1.20023040e+02])
vars = np.array([2.81408733e-04, 2.66845279e+04, 1.49823072e+03, 2.58834946e-01,
        9.33931029e-01, 2.00304368e+00, 1.42230878e-01, 1.52871978e+00,
        2.78359637e-01, 5.08976571e-01, 1.96979357e+00, 2.34725629e+01,
        1.65692805e-03, 3.28087242e-05, 1.11131253e+02])
names = ['SG', 'Pressure', 'Temp', 'C1', 'C2', 'C3', 'iC4', 'C4', 'iC5', 'C5', 'C6', 'C7', 'CO2', 'N2', 'MW']

#equations for scaling + unscaling data
def scale_data(array,means=means,stds=vars**0.5):
    return (array-means)/stds

def unscale_data(Trans_data, means=means, stds=vars**0.5):
    return Trans_data*stds+means



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    file = request.files['inputFile']
    Input = pd.read_csv(file)

    Input_value = Input.drop(['Sample Name', 'Sample Number'], axis=1)
    Scaled_Input = scale_data(Input_value.values)
    Input_df2 = pd.DataFrame(Scaled_Input,columns = names)
    Input_df3 = Input_df2.drop(['MW','N2','Pressure','Temp','CO2','iC5'],axis=1)
    Prediction = MW_Model.predict(Input_df3.values)

    rows = len(Input_df3)
    #dataframe of zeros (x_rows, x_columns)
    Zeros = pd.DataFrame(np.zeros((rows, 15)),columns = names)
    Zeros2 = Zeros.drop(['MW'],axis=1)
    Zeros2.insert(14,"MW",Prediction)
    MW_Prediction = unscale_data(Zeros2.values)
    MW_DF = pd.DataFrame(MW_Prediction,columns = names)


    MW_DF = MW_DF.pop('MW')
    Sample_name = Input.pop('Sample Name')
    Sample_number = Input.pop('Sample Number')
    output = pd.concat([Sample_name, Sample_number, MW_DF,], axis=1, sort=False)

    return render_template('index.html', table1 = output.to_html(header = 'true'))


if __name__ == "__main__":
    app.run(debug=True)

