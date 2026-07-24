from flask import Flask, render_template, request
import joblib
import pandas as pd

app = flask(__name__)

model = joblib.load("models/pipe_ohe.pkl")
