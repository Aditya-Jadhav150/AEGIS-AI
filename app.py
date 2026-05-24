import warnings
import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from facenet_pytorch import MTCNN
from datetime import datetime, timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from collections import defaultdict
import time

# Core components for the Fusion Engine
from core.alignment import GeometricAligner
from core.diffusion_latent import DiffusionErrorLoop
from core.statistical_extraction import StatisticalFeatureExtractor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-super-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Simple memory cache for rate limiting IPs (tracks failed attempts only)
failed_logins = defaultdict(list)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# CRITICAL HF FIX: Allow cookies to survive inside across cross-origin iframes!
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'None'
app.config['REMEMBER_COOKIE_SECURE'] = True

# Tell Flask it is behind a proxy (like Hugging Face) so Secure=True works over HTTP
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password_hash = db.Column(db.String(300), nullable=True) # Now nullable for Google users
    google_id = db.Column(db.String(150), unique=True, nullable=True)
    last_username_change = db.Column(db.DateTime, nullable=True)
    ai_data_optin = db.Column(db.Boolean, default=False, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Guarantee the SQLite User database exists if deployed out via WSGI Container (Docker)
with app.app_context():
    try:
        from sqlalchemy import text
        db.session.execute(text('ALTER TABLE user ADD COLUMN ai_data_optin BOOLEAN DEFAULT 0'))
        db.session.commit()
    except Exception:
        pass
    try:
        db.create_all()
    except Exception as e:
        print(f"Skipping database creation (likely handled by another worker): {e}")

# --- PyTorch & Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing Aegis-AI forensic extraction pipelines on device:", device)

# Initialize Fusion Engine core components globally for speed
fusion_model = None
fusion_scaler = None
aligner = None
error_loop = None
stat_extractor = None

try:
    model_path = os.path.join('dataset', 'fusion_engine_best.pkl')
    scaler_path = os.path.join('dataset', 'scaler.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        fusion_model = joblib.load(model_path)
        fusion_scaler = joblib.load(scaler_path)
        print("🗄️  XGBoost Fusion Model and Scaler loaded successfully.")
    else:
        print("⚠️  Warning: Model/Scaler pkl files not found. Please train first.")
        
    aligner = GeometricAligner(device=device)
    error_loop = DiffusionErrorLoop(device=device)
    stat_extractor = StatisticalFeatureExtractor()
    print("🧬  Sub-network extractors successfully mounted on GPU/CPU.")
except Exception as e:
    print(f"❌  ERROR during pipeline initialization: {e}")

# ----- Forensic Helper Functions -----
def high_freq_energy(freq_tensor: torch.Tensor) -> float:
    freq = freq_tensor.squeeze().cpu()
    h, w = freq.shape
    mask = torch.zeros_like(freq, dtype=torch.bool)
    margin_h = h // 4
    margin_w = w // 4
    mask[:margin_h, :] = True
    mask[-margin_h:, :] = True
    mask[:, :margin_w] = True
    mask[:, -margin_w:] = True
    return float(torch.sum(torch.abs(freq)[mask]))

def compute_entropy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(shannon_entropy(gray))

def compute_edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)

def compute_laplacian_variance(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def compute_color_kurtosis(img_rgb):
    ks = [kurtosis(img_rgb[..., c].ravel()) for c in range(3)]
    return float(np.mean(ks))

def compute_jpeg_consistency(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct_full = cv2.dct(gray)
    mask = np.ones_like(dct_full, dtype=bool)
    mask[:8, :8] = False
    return float(np.var(dct_full[mask]))

def predict_image(image_path):
    if fusion_model is None or fusion_scaler is None:
        return {"success": False, "error": "XGBoost Fusion Engine not initialized. Re-train the model first."}
        
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return {"success": False, "error": f"Could not read image: {image_path}"}
            
        # 1. Align & Crop Face
        aligned = aligner.align_and_crop(bgr, return_tensor=True)
        if aligned is None:
            return {"success": False, "error": "Face detection failed - no face found in the image."}
            
        # Save the aligned cropped face to disk to serve it to the frontend
        filename = os.path.basename(image_path)
        aligned_filename = "aligned_" + filename
        aligned_filepath = os.path.join(os.path.dirname(image_path), aligned_filename)
        
        # Crop is RGB, convert to BGR for OpenCV save
        aligned_np = (aligned.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        aligned_bgr = cv2.cvtColor(aligned_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aligned_filepath, aligned_bgr)
            
        # 2. Extract features
        spatial_score = float(torch.mean(torch.abs(aligned)).item())
        
        # Frequency score (exactly matching data_pipeline.py on the normalized aligned tensor)
        gray_tensor = 0.2989 * aligned[0:1, :, :] + 0.5870 * aligned[1:2, :, :] + 0.1140 * aligned[2:3, :, :]
        gray_tensor = gray_tensor.unsqueeze(0)  # Shape: [1, 1, 512, 512]
        freq_complex = torch.fft.fft2(gray_tensor)
        freq_shifted = torch.fft.fftshift(torch.abs(freq_complex), dim=(-2, -1))
        freq_tensor = torch.log(1 + freq_shifted)
        freq_score = high_freq_energy(freq_tensor)
        
        # Latent score
        latent_err = error_loop(aligned.unsqueeze(0))
        latent_score = float(torch.mean(torch.abs(latent_err)).item())
        
        # Statistical score
        stat_tensor = stat_extractor(aligned.unsqueeze(0)).cpu()
        stat_score = float(torch.mean(stat_tensor).item())
        
        # Conversions and statistics
        entropy_score = compute_entropy(aligned_np)
        edge_density_score = compute_edge_density(aligned_np)
        laplacian_var_score = compute_laplacian_variance(aligned_np)
        color_kurtosis_score = compute_color_kurtosis(aligned_np)
        jpeg_consistency_score = compute_jpeg_consistency(aligned_np)
        
        # 3. Assemble features
        feature_dict = {
            "spatial_score": spatial_score,
            "freq_score": freq_score,
            "latent_score": latent_score,
            "stat_score": stat_score,
            "entropy": entropy_score,
            "edge_density": edge_density_score,
            "laplacian_variance": laplacian_var_score,
            "color_kurtosis": color_kurtosis_score,
            "jpeg_consistency": jpeg_consistency_score,
        }
        df_feat = pd.DataFrame([feature_dict])
        df_scaled = fusion_scaler.transform(df_feat)
        
        # 4. Predict
        prob_fake = float(fusion_model.predict_proba(df_scaled)[0, 1])
        prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = prob_fake * 100 if prob_fake >= 0.5 else (1 - prob_fake) * 100
        
        return {
            "success": True,
            "prediction": prediction,
            "fake_prob": round(prob_fake * 100, 2),
            "real_prob": round((1 - prob_fake) * 100, 2),
            "confidence": round(confidence, 2),
            "aligned_filename": aligned_filename,
            "features": {
                "spatial_score": round(spatial_score, 4),
                "freq_score": round(freq_score, 2),
                "latent_score": round(latent_score, 4),
                "stat_score": round(stat_score, 4),
                "entropy": round(entropy_score, 4),
                "edge_density": round(edge_density_score, 4),
                "laplacian_variance": round(laplacian_var_score, 2),
                "color_kurtosis": round(color_kurtosis_score, 4),
                "jpeg_consistency": round(jpeg_consistency_score, 2)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Web Routes ---
@app.route('/aegis-override-system')
@login_required
def admin():
    if not current_user.email or current_user.email.strip().lower() != 'adityajadhav300405@gmail.com':
        return redirect(url_for('index'))
    return render_template('admin.html', user=current_user)

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/login', methods=['GET'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    ip = request.remote_addr or request.headers.get('X-Forwarded-For', 'unknown-ip')
    now = datetime.utcnow()
    
    # Prune failures older than 5 hours (18000 seconds)
    failed_logins[ip] = [t for t in failed_logins[ip] if (now - t).total_seconds() < 18000]
    
    if len(failed_logins[ip]) >= 5:
        return jsonify({"success": False, "message": "CRITICAL: Too many failed attempts. Your device is locked out of logins for 5 hours."}), 429

    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    # Google users will not have a password hash, explicitly block password login for them if hash is None
    if user and user.password_hash and check_password_hash(user.password_hash, data.get('password')):
        login_user(user, remember=True)
        # Clear failures on successful login
        if ip in failed_logins:
            del failed_logins[ip]
        return jsonify({"success": True})
        
    # Record the failure
    failed_logins[ip].append(now)
    attempts_left = 5 - len(failed_logins[ip])
    return jsonify({"success": False, "message": f"Invalid username or password. {attempts_left} attempts remaining."}), 401

@app.route('/api/auth/google', methods=['POST'])
def api_google_login():
    data = request.json
    token = data.get('credential')
    client_id = data.get('clientId')
    
    if not token or not client_id:
        return jsonify({"success": False, "message": "Missing Google payload"}), 400
        
    try:
        # Validate the JWT natively using Google's Python SDK
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
        
        google_id = idinfo["sub"]
        email = idinfo.get("email")
        # Base username strategy from email prefix
        base_username = email.split('@')[0] if email else f"User{google_id[:6]}"
        
        user = User.query.filter_by(google_id=google_id).first()
        
        if not user:
            # Handle potential username collisions automatically during first creation
            candidate = base_username
            attempt = 1
            while User.query.filter_by(username=candidate).first():
                candidate = f"{base_username}{attempt}"
                attempt += 1

            user = User(
                username=candidate,
                email=email,
                google_id=google_id,
                # Force last_username_change to 7 days ago initially so they can change auto-generated names immediately
                last_username_change=datetime.utcnow() - timedelta(days=8)
            )
            db.session.add(user)
            db.session.commit()
            
        login_user(user, remember=True)
        return jsonify({"success": True})
    except ValueError:
        return jsonify({"success": False, "message": "Invalid Google token"}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or len(username) < 5:
        return jsonify({"success": False, "message": "Username must be at least 5 characters long."})
        
    if not password or len(password) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters long."})
    if not re.search(r"[a-z]", password):
        return jsonify({"success": False, "message": "Password must contain at least one lowercase letter."})
    if not re.search(r"[A-Z]", password):
        return jsonify({"success": False, "message": "Password must contain at least one uppercase letter."})
    if not re.search(r"[0-9]", password):
        return jsonify({"success": False, "message": "Password must contain at least one number."})
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return jsonify({"success": False, "message": "Password must contain at least one special character."})
    
    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "message": "Username already exists. Please choose a new combination."})
    
    new_user = User(
        username=username, 
        password_hash=generate_password_hash(password),
        last_username_change=datetime.utcnow() # Lock them for 7 days upon standard creation
    )
    db.session.add(new_user)
    db.session.commit()
    
    # Auto-login after registration
    login_user(new_user)
    return jsonify({"success": True})

@app.route('/api/admin/users', methods=['GET'])
@login_required
def api_admin_users():
    if not current_user.email or current_user.email.strip().lower() != 'adityajadhav300405@gmail.com':
        return jsonify({"success": False, "message": "FORBIDDEN: Admin access only."}), 403
    
    users = User.query.all()
    user_data = []
    for u in users:
        auth_type = "Google" if u.google_id else "Password"
        user_data.append({
            "id": u.id,
            "username": u.username,
            "email": u.email or "Unassigned",
            "auth_type": auth_type,
            "last_username_change": u.last_username_change.isoformat() if u.last_username_change else None,
            "ai_data_optin": u.ai_data_optin
        })
        
    return jsonify({
        "success": True,
        "users": user_data
    })

@app.route('/api/me', methods=['GET'])
@login_required
def api_me():
    last_change = current_user.last_username_change
    days_since_change = (datetime.utcnow() - last_change).days if last_change else 999
    is_locked = days_since_change < 7
    days_remaining = max(0, 7 - days_since_change)
    
    return jsonify({
        "success": True,
        "user": {
            "username": current_user.username,
            "email": current_user.email,
            "is_locked": is_locked,
            "days_remaining": days_remaining,
            "is_google": current_user.google_id is not None
        }
    })

@app.route('/api/update_username', methods=['POST'])
@login_required
def api_update_username():
    data = request.json
    new_username = data.get('new_username')
    
    if not new_username or len(new_username) < 5:
        return jsonify({"success": False, "message": "Username must be at least 5 characters long."})
        
    if current_user.username == new_username:
         return jsonify({"success": False, "message": "This is already your username."})
         
    # Enforce 7 day lockout
    if current_user.last_username_change:
        days_since_change = (datetime.utcnow() - current_user.last_username_change).days
        if days_since_change < 7:
            return jsonify({"success": False, "message": f"You changed your username recently. Please wait {7 - days_since_change} more days."})

    # Ensure absolute uniqueness
    if User.query.filter_by(username=new_username).first():
        return jsonify({"success": False, "message": "Username already exists. Please choose a new combination."})
        
    current_user.username = new_username
    current_user.last_username_change = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "message": "Username updated successfully."})

@app.route('/api/update_optin', methods=['POST'])
@login_required
def api_update_optin():
    data = request.json
    optin_status = data.get('ai_data_optin')
    if optin_status is not None:
        current_user.ai_data_optin = bool(optin_status)
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file chunk found."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected."}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        if result["success"]:
            # Expose public visual path
            result["image_url"] = f"/static/uploads/{filename}"
            if "aligned_filename" in result:
                result["aligned_url"] = f"/static/uploads/{result['aligned_filename']}"
            return jsonify(result)
        else:
            return jsonify({"success": False, "message": result.get("error")}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
