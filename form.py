from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired



class SearchForm(FlaskForm):
    query = StringField("Search an ML topic", validators=[DataRequired()])
    submit = SubmitField("Submit")