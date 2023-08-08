from app import app  # Import your Flask app instance
from gunicorn.app.base import Application

class FlaskApp(Application):
    def init(self, parser, opts, args):
        return {
            'bind': '0.0.0.0:8000',  # Replace with the desired host and port
        }

    def load(self):
        return app

if __name__ == '__main__':
    FlaskApp().run()
