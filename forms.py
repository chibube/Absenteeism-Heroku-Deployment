from flask_wtf import FlaskForm
from wtforms import SubmitField, IntegerField, DateField, SelectField
from wtforms.validators import DataRequired, InputRequired

#creating a class that inherits the flaskform and defines the list of the fields in the form
class EmployeeForm(FlaskForm):
    id = IntegerField("Employee ID", validators=[DataRequired()])
    reason_for_absence = IntegerField("Reason for absence", validators=[InputRequired()])
    date = DateField("Date", validators=[DataRequired()], format ='%d/%m/%Y',  description='dd/mm/yyyy')
    transport = IntegerField("Transport expense", validators=[DataRequired()])
    distance = IntegerField("Distance to work", validators=[DataRequired()])
    age = IntegerField("Age", validators=[DataRequired()])
    work_load = IntegerField("Average Daily Workload", validators=[DataRequired()])
    bmi = IntegerField("Body Mass Index", validators=[DataRequired()])
    education = SelectField("Education", validators=[DataRequired()], choices=[(1, 'High School'), (2, 'Graduate'),
                                                                               (3, 'Postgraduate'), (4, 'Masters or PhD')])
    children = IntegerField("Number of children", validators=[InputRequired()])
    pets = IntegerField("Number of Pets", validators=[InputRequired()])
    submit = SubmitField('Submit')