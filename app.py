from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from forms import EmployeeForm
import pandas as pd
from model import absenteeism_model

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'hard string'


predictor = absenteeism_model('model', 'scaler')
#the home page will have both a get and post method to enable us render the template before and after submission
#upon validation of the form we will save the data in the forms and create a dictionary which will be converted
#into pandas dataframe and passed into our classes for preprocessing and prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    id = None
    form = EmployeeForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            id = form.id.data
            reason_for_absence = form.reason_for_absence.data
            date = form.date.data
            transport = form.transport.data
            distance = form.distance.data
            age = form.distance.data
            work_load = form.work_load.data
            bmi = form.bmi.data
            education = form.education.data
            children = form.children.data
            pets = form.pets.data

            df = {'ID':[id], 'Reason for Absence':[reason_for_absence], 'Date':[date], 'Transportation Expense':[transport],
           'Distance to Work': [distance], 'Age':[age], 'Daily Work Load Average': [work_load], 'Body Mass Index':[bmi],
           'Education': [education], 'Children': [children], 'Pets': [pets]}


            df = pd.DataFrame.from_dict(df)
            predictor.load_and_clean_data(df)
            result = predictor.predicted_outputs()
            probability = round(result['Probability'][0] * 100, 1)
            prediction = result['Prediction'][0]

    return render_template('index.html', form=form, prediction = prediction, probability=probability, id=id)



if __name__ == '__main__':
    app.run()
