from flask import Flask

app = Flask(__name__)

#from current app package(dir) import views.py
from app import views 
