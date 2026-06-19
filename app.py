import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename
from llm_model import generate_chat_details, LLM_AVAILABLE, DB_AVAILABLE, CLIP_MODEL_AVAILABLE

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/status")
def status():
    # Returns status of the backend models
    from vgg16pred import VGG_AVAILABLE
    from img2txt import BLIP_AVAILABLE
    return jsonify({
        "llama2": "Active (Local)" if LLM_AVAILABLE else "Mock/Simulation Mode",
        "faiss": "Active (Local)" if DB_AVAILABLE else "Mock/Simulation Mode",
        "clip": "Active (Local)" if CLIP_MODEL_AVAILABLE else "Mock/Simulation Mode",
        "vgg16": "Active (Local)" if VGG_AVAILABLE else "Mock/Simulation Mode",
        "blip": "Active (Local)" if BLIP_AVAILABLE else "Mock/Simulation Mode"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        text = request.form.get("text", "").strip()
        image_file = request.files.get("image")
        
        if not text and not image_file:
            return jsonify({"error": "No query text or image provided."}), 400

        image_path = None
        image_url = None
        pil_image = None

        if image_file and image_file.filename != '':
            # Generate a unique secure filename to prevent cache conflicts
            ext = os.path.splitext(image_file.filename)[1]
            if not ext:
                ext = ".jpg"
            filename = secure_filename(f"img_{int(time.time())}{ext}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            
            # Load as PIL image for model processing
            pil_image = Image.open(image_path)
            # URL to access the uploaded file from browser
            image_url = f"/uploads/{filename}"

        # Run models
        result = generate_chat_details(text, pil_image)
        
        # Add the uploaded image URL to the result response
        result["image_url"] = image_url
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8000, host="0.0.0.0")

