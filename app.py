  
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

#Pipeline for MW calcs
def compute_MW(SG, P, T, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, CO2, N2, MW):
#    data = np.loadtxt(os.path.join('uploads', filename))
        
    MW_Model = joblib.load('MW_model_initial.pk1')
    #Data = pd.read_csv('Combined Flowcals2.csv')
         
    #Data_scaled = (Data).astype(float)
    #scal = StandardScaler()
    #Data_scaled = scal.fit_transform(Data_scaled)
    #C = list(Data)
    #names = list(Data.columns)
    #Data_scaled = pd.DataFrame(Data_scaled,columns = names)


    #Input_test = [[0.78, 400, 60, 0.5, 1.5, 2.1, 1.0, 3.5, 2.8, 4.2, 8.7, 74, 0.3, 0.002, 0]]
    MW = 0 
    Input_test = [[SG, P, T, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, CO2, N2, MW]]
    Input = pd.DataFrame(Input_test, columns = names)

        
    Scaled_Input = scale_data(Input.values)
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

    #check the model to validate values
    output = MW_DF.pop('MW').values
    print(output)


    return output




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = sum(final_features)
    html_input = [float(x) for x in request.form.values()]

    #output = round(prediction[0], 4)

    SG = html_input[0]
    P = html_input[1]
    T = html_input[2]
    C1 = html_input[3]
    C2 = html_input[4]
    C3 = html_input[5]
    iC4 = html_input[6]
    C4 = html_input[7]
    iC5 = html_input[8]
    C5 = html_input[9]
    C6 = html_input[10]
    C7 = html_input[11]
    CO2 = html_input[12]
    N2 = html_input[13]
    MW = 0
    
    #calc = 5954/(((141.5/html_input[0])-131.5)-8.811)
    result = compute_MW(SG, P, T, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, CO2, N2, MW)
    output = round(result[0],2)    
    output_2 = 5954/(((141.5/html_input[0])-131.5)-8.811)
    output_3 = round(0.5*output + 0.5*output_2,2)

    return render_template('index.html', prediction_text='    Molecular Weight= {}'.format(output_3))


if __name__ == "__main__":
    app.run(debug=True)
