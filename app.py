import warnings
import os
from dotenv import load_dotenv

load_dotenv()
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

# Core components
from core.alignment import GeometricAligner
from core.diffusion_latent import DiffusionErrorLoop
from core.statistical_extraction import StatisticalFeatureExtractor
from core.metrics import ForensicMetricExtractor
from core.metadata_forensics import MetadataForensicsEngine
from core.video_pipeline import VideoPipeline
from core.deepfake_detector import DeepfakeDetector

# Primary deepfake detector (lazy-loaded on first prediction)
deepfake_detector = DeepfakeDetector(device='cpu')
# Video pipeline (uses Haar cascade + ViT internally)
video_pipeline = VideoPipeline(device='cpu')
# Metadata forensics engine
metadata_engine = MetadataForensicsEngine()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-hf-space-secret-key')
# Configure Database (Cloud PostgreSQL priority, SQLite fallback)
db_url = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url

# Flask-WTF CSRF Protection
csrf = CSRFProtect(app)

# Flask-Limiter setup
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

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
    is_admin = db.Column(db.Boolean, default=False, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Guarantee the SQLite User database exists if deployed out via WSGI Container (Docker)
with app.app_context():
    try:
        from sqlalchemy import text
        db.session.execute(text('ALTER TABLE user ADD COLUMN ai_data_optin BOOLEAN DEFAULT 0'))
        db.session.execute(text('ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0'))
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
metric_extractor = None

try:
    model_path = os.path.join('dataset', 'fusion_engine_best.json')
    scaler_path = os.path.join('dataset', 'scaler.json')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        from xgboost import XGBClassifier
        import json
        
        # Load XGBoost model from JSON
        fusion_model = XGBClassifier()
        fusion_model.load_model(model_path)
        
        # Reconstruct StandardScaler from JSON parameters
        fusion_scaler = StandardScaler()
        with open(scaler_path, 'r') as f:
            s_data = json.load(f)
        fusion_scaler.mean_ = np.array(s_data["mean"])
        fusion_scaler.var_ = np.array(s_data["var"])
        fusion_scaler.scale_ = np.array(s_data["scale"])
        fusion_scaler.n_features_in_ = s_data["n_features_in"]
        
        print("XGBoost Fusion Model and Scaler loaded from plain-text JSON successfully.")
    else:
        print("Warning: Model/Scaler JSON files not found. Please train first.")
        
    aligner = GeometricAligner(device=device)
    error_loop = DiffusionErrorLoop(device=device)
    stat_extractor = StatisticalFeatureExtractor()
    metric_extractor = ForensicMetricExtractor(device=device)
    print("Sub-network extractors successfully mounted on GPU/CPU.")
except Exception as e:
    print(f"ERROR during pipeline initialization: {e}")
    raise e

def predict_image(image_path):
    """
    Primary image prediction pipeline.
    
    1. Run the ViT deepfake detector as the primary verdict engine.
    2. In parallel, run RetinaFace alignment + 9 forensic metrics for dashboard.
    3. If face alignment fails, still return the ViT verdict.
    """
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return {"success": False, "error": f"Could not read image: {image_path}"}

        # ── Primary verdict: ViT deepfake detector ───────────────────────────────
        primary = deepfake_detector.predict_path(image_path)
        prediction  = primary["prediction"]
        fake_prob   = primary["fake_prob"]
        real_prob   = primary["real_prob"]
        confidence  = primary["confidence"]

        result = {
            "success":    True,
            "prediction": prediction,
            "fake_prob":  fake_prob,
            "real_prob":  real_prob,
            "confidence": confidence,
            "method":     "deepfake_vit",
            "features":   {},
        }

        # ── Supplementary: Metadata forensics ───────────────────────────────────
        try:
            meta = metadata_engine.analyze(image_path)
            result["features"]["metadata_forensic_score"] = round(
                meta.get("metadata_forensic_score", 0.5), 4
            )
        except Exception:
            pass

        # ── Supplementary: RetinaFace alignment + forensic metrics ───────────────
        # (dashboard visualisation only — does NOT affect the verdict)
        try:
            aligned = aligner.align_and_crop(bgr, return_tensor=True)
            if aligned is not None:
                filename = os.path.basename(image_path)
                aligned_filename = "aligned_" + filename
                aligned_filepath = os.path.join(os.path.dirname(image_path), aligned_filename)

                mean_v = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std_v  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                aligned_np  = aligned.cpu().numpy()
                unnorm      = (aligned_np * std_v + mean_v) * 255.0
                aligned_rgb = np.clip(unnorm, 0, 255).transpose(1, 2, 0).astype(np.uint8)
                cv2.imwrite(aligned_filepath, cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
                result["aligned_filename"] = aligned_filename

                fdict = metric_extractor.extract_all(aligned, aligned_rgb)
                result["features"].update({
                    "spatial_score":      round(fdict["spatial_score"], 4),
                    "freq_score":         round(fdict["freq_score"], 2),
                    "latent_score":       round(fdict["latent_score"], 4),
                    "stat_score":         round(fdict["stat_score"], 4),
                    "entropy":            round(fdict["entropy"], 4),
                    "edge_density":       round(fdict["edge_density"], 4),
                    "laplacian_variance": round(fdict["laplacian_variance"], 2),
                    "color_kurtosis":     round(fdict["color_kurtosis"], 4),
                    "jpeg_consistency":   round(fdict["jpeg_consistency"], 2),
                })
        except Exception:
            pass  # Forensic metrics are optional — verdict already set

        return result

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
@limiter.limit("5 per 5 hours")
def api_login():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    # Google users will not have a password hash, explicitly block password login for them if hash is None
    if user and user.password_hash and check_password_hash(user.password_hash, data.get('password')):
        login_user(user, remember=True)
        return jsonify({"success": True})
        
    return jsonify({"success": False, "message": "Invalid username or password."}), 401

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
            "is_google": current_user.google_id is not None,
            "ai_data_optin": current_user.ai_data_optin
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
        IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
        VIDEO_EXTENSIONS = {'mp4', 'webm', 'mov', 'avi'}
        ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS)
        
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "message": "Invalid file type. Upload images or videos only."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if ext in VIDEO_EXTENSIONS:
            # Route to Video Pipeline (Haar cascade + ViT deepfake detector)
            try:
                vid_result = video_pipeline.process_video(filepath, fps_extraction=1)
                if not vid_result:
                    return jsonify({"success": False, "message": "Could not analyse video — no frames could be extracted."}), 400

                result = {
                    "success":         True,
                    "prediction":      vid_result["prediction"],
                    "fake_prob":       vid_result["fake_prob"],
                    "real_prob":       vid_result["real_prob"],
                    "confidence":      vid_result["confidence"],
                    "method":          vid_result.get("method", "video_deepfake_vit"),
                    "frames_analysed": vid_result.get("frames_analysed", 0),
                    "faces_found":     vid_result.get("faces_found", 0),
                    "frame_scores":    vid_result.get("frame_scores", []),
                    "features":        {},
                    "image_url":       f"/static/uploads/{filename}",
                    "is_video":        True,
                }
            except Exception as e:
                return jsonify({"success": False, "message": f"Video processing error: {e}"}), 500
        else:
            # Route to Image Pipeline
            result = predict_image(filepath)
            
        if result["success"]:
            if not result.get("image_url"):
                result["image_url"] = f"/static/uploads/{filename}"
            if "aligned_filename" in result:
                result["aligned_url"] = f"/static/uploads/{result['aligned_filename']}"
            return jsonify(result)
        else:
            return jsonify({"success": False, "message": result.get("error")}), 500

def send_deletion_email(recipient_email, username, reason):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    try:
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user or "noreply@aegis-ai.com")

    subject = "AEGIS-AI Account Deletion Notification"
    body = f"""Dear {username},

This is an automated notification to inform you that your operator account on AEGIS-AI has been deleted by an administrator.

Reason for Deletion:
{reason}

If you believe this was in error, please contact your system administrator.

Best regards,
AEGIS-AI Security Core
"""

    print(f"--- SIMULATED EMAIL TO {recipient_email} ---")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print("-----------------------------------------")

    if not smtp_user or not smtp_password:
        print("SMTP credentials not configured. Email simulation completed.")
        return True

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_from
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_from, recipient_email, msg.as_string())
        server.close()
        print("Email sent successfully via SMTP.")
        return True
    except Exception as e:
        print(f"Failed to send email via SMTP: {e}")
        return False

@app.route('/api/admin/delete_user', methods=['POST'])
@login_required
def api_admin_delete_user():
    # Admin verification: match the specified email address
    if not current_user.email or current_user.email.strip().lower() != 'adityajadhav300405@gmail.com':
        return jsonify({"success": False, "message": "FORBIDDEN: Admin access only."}), 403

    data = request.json or {}
    user_id = data.get('user_id')
    reason = data.get('reason', 'No reason specified by administration.')

    if not user_id:
        return jsonify({"success": False, "message": "Missing user ID."}), 400

    user_to_delete = User.query.get(user_id)
    if not user_to_delete:
        return jsonify({"success": False, "message": "User not found."}), 404

    if user_to_delete.id == current_user.id:
        return jsonify({"success": False, "message": "Self-destruction blocked. You cannot delete your own admin account."}), 400

    username = user_to_delete.username
    recipient_email = user_to_delete.email

    try:
        db.session.delete(user_to_delete)
        db.session.commit()

        # Send/Log notification email if user has a valid email
        if recipient_email and recipient_email != "Unassigned":
            send_deletion_email(recipient_email, username, reason)

        return jsonify({"success": True, "message": f"Operator '{username}' deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Database write error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
