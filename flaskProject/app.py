
from flask import Flask, send_file, request
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from flask_cors import CORS
import io
import json

app = Flask(__name__)
CORS(app)
def generate_image(A, B, C, fx, fy, gx, gy, hx, hy, noise):
    x_res, y_res = 200, 100
    x = np.linspace(0, 5 * np.pi, x_res)
    y = np.linspace(0, 5 * np.pi, y_res)
    X, Y = np.meshgrid(x, y)

    Z = (A * np.sin(fx * X) * np.cos(fy * Y) +
         B * np.sin(gx * X**2 + gy * Y**2) +
         C * np.sin(hx * X**3 + hy * Y**3))

    random_noise = np.random.randn(*Z.shape) * noise
    random_noise = gaussian_filter(random_noise, sigma=4)
    Z += random_noise

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='viridis')])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 5 * np.pi]),
            yaxis=dict(range=[0, 5 * np.pi]),
            zaxis=dict(showgrid=False, showticklabels=False)
        ),
        scene_camera=dict(eye=dict(x=-1, y=-1, z=np.max(Z)))

    )


# Speichern als statisches Bild
    img_io = io.BytesIO()
    fig.write_image(img_io, format='png', scale=2)
    img_io.seek(0)
    return img_io


@app.route('/sinwave', methods=['POST'])
def generate_img():
    try:
        data = request.get_json()
        A = float(data.get('A', 1.3))
        B = float(data.get('B', 0.5))
        C = float(data.get('C', 0.3))
        fx = float(data.get('fx', 1.5))
        fy = float(data.get('fy', 1.3))
        gx = float(data.get('gx', 0.6))
        gy = float(data.get('gy', 0.3))
        hx = float(data.get('hx', 0.01))
        hy = float(data.get('hy', 0.01))
        noise = float(data.get('noise', 0))
    except Exception as e:
        return {"error": "Invalid input data", "message": str(e)}, 400

    return send_file(generate_image(A, B, C, fx, fy, gx, gy, hx, hy, noise), mimetype='image/webp')



if __name__ == '__main__':
    app.run(debug=True)
