from flask import Flask

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

from app import routes  # Import routes here

# Optionally, you can add more configuration or extensions here

if __name__ == '__main__':
    app.run(debug=True)
