from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
with open('model_21-03-2024-23-28-32-039.pkl', 'rb') as file:
    model = pickle.load(file)

if __name__ == '__main__':
    app.run(debug=True)