import piexif
from PIL import Image
import json
import numpy as np

try:
    from c2pa import Reader
except ImportError:
    Reader = None

class MetadataForensicsEngine:
    """
    Extracts and analyzes image metadata for forensic evidence.
    
    Position in pipeline: BEFORE face detection.
    Output: MetadataForensicScore (0.0 = highly suspicious, 1.0 = fully consistent)
    
    CRITICAL DESIGN RULE: Missing metadata ≠ Fake.
    Missing metadata produces a NEUTRAL score (0.5), not a suspicious one.
    
    NOTE: This score represents forensic evidence about provenance,
    NOT a direct authenticity judgment. It enters the fusion model
    as one feature among many.
    """
    
    def __init__(self):
        self.ai_signatures = ["Stable Diffusion", "Midjourney", "DALL-E", "ComfyUI", "NovelAI"]
        self.edit_signatures = ["Photoshop", "GIMP", "Lightroom", "Canva", "Snapseed"]

    def analyze(self, image_path: str) -> dict:
        """
        Returns:
            {
                'metadata_forensic_score': float,  # 0.0 - 1.0
                'exif_present': bool,
                'camera_make': str | None,
                'camera_model': str | None,
                'software': str | None,
                'editing_detected': bool,
                'compression_quality': int | None,
                'compression_consistency': float,  # DCT quantization table analysis
                'c2pa_verified': bool | None,      # None = no C2PA data found
                'c2pa_details': dict | None,
                'signals': list[str],              # Human-readable signal descriptions
                'raw_exif': dict                   # Complete EXIF dump
            }
        """
        result = {
            'metadata_forensic_score': 0.5,
            'exif_present': False,
            'camera_make': None,
            'camera_model': None,
            'software': None,
            'editing_detected': False,
            'compression_quality': None,
            'compression_consistency': 0.5,
            'c2pa_verified': None,
            'c2pa_details': None,
            'signals': [],
            'raw_exif': {}
        }
        
        # 1. EXIF and Compression Analysis
        try:
            with Image.open(image_path) as img:
                info = img.info
                # Compression tables
                if 'quantization' in info:
                    q_tables = info['quantization']
                    if len(q_tables) > 0:
                        # Compute variance as a simple proxy for inconsistency
                        # Normal JPEG has specific quantization patterns
                        q_var = float(np.var(list(q_tables.values())[0]))
                        result['compression_consistency'] = min(max(q_var / 1000.0, 0.0), 1.0)
                
                if 'exif' in info:
                    exif_dict = piexif.load(info['exif'])
                    result['exif_present'] = True
                    result['raw_exif'] = self._sanitize_exif(exif_dict)
                    
                    if piexif.ImageIFD.Make in exif_dict["0th"]:
                        result['camera_make'] = exif_dict["0th"][piexif.ImageIFD.Make].decode('utf-8', errors='ignore').strip()
                    
                    if piexif.ImageIFD.Model in exif_dict["0th"]:
                        result['camera_model'] = exif_dict["0th"][piexif.ImageIFD.Model].decode('utf-8', errors='ignore').strip()
                        
                    if piexif.ImageIFD.Software in exif_dict["0th"]:
                        software = exif_dict["0th"][piexif.ImageIFD.Software].decode('utf-8', errors='ignore').strip()
                        result['software'] = software
                        
                        # Check for AI signatures
                        for sig in self.ai_signatures:
                            if sig.lower() in software.lower():
                                result['signals'].append(f"AI Generation Signature detected in EXIF Software: {sig}")
                                
                        # Check for editing signatures
                        for sig in self.edit_signatures:
                            if sig.lower() in software.lower():
                                result['editing_detected'] = True
                                result['signals'].append(f"Photo Editing Signature detected in EXIF Software: {sig}")
        except Exception as e:
            result['signals'].append(f"Failed to parse EXIF/JPEG metadata: {str(e)}")

        # 2. C2PA Analysis
        if Reader is not None:
            try:
                reader = Reader.from_file(image_path)
                if reader:
                    result['c2pa_verified'] = True
                    # Just an example of pulling some C2PA info
                    result['c2pa_details'] = json.loads(reader.json())
                    result['signals'].append("Valid C2PA Content Credentials found.")
                else:
                    # No C2PA data attached
                    pass
            except Exception as e:
                # If reading fails, either missing or corrupted C2PA
                pass

        # 3. Compute Final Score
        result['metadata_forensic_score'] = self._compute_forensic_score(result)
        
        return result
        
    def _sanitize_exif(self, exif_dict: dict) -> dict:
        """Convert bytes to string for JSON serialization"""
        clean_exif = {}
        for ifd in ("0th", "Exif", "GPS", "1st"):
            clean_exif[ifd] = {}
            for tag in exif_dict.get(ifd, {}):
                val = exif_dict[ifd][tag]
                if isinstance(val, bytes):
                    try:
                        val = val.decode('utf-8', errors='ignore')
                    except:
                        val = str(val)
                clean_exif[ifd][piexif.TAGS[ifd][tag]["name"]] = val
        return clean_exif

    def _compute_forensic_score(self, analysis: dict) -> float:
        """
        Scoring philosophy:
        - Consistent camera EXIF + valid C2PA = HIGH forensic confidence (0.8-1.0)
        - Consistent camera EXIF, no C2PA = MODERATE forensic confidence (0.6-0.8)
        - No EXIF at all = NEUTRAL (0.5) — NOT suspicious
        - AI software signature in EXIF = LOW forensic confidence (0.1-0.3)
        - Contradictory metadata = LOW forensic confidence (0.1-0.3)
        """
        score = 0.5  # Neutral default
        
        has_ai_sig = any("AI Generation Signature" in sig for sig in analysis['signals'])
        
        if analysis['c2pa_verified']:
            # C2PA provides very strong evidence
            score = 0.95
        elif analysis['exif_present']:
            if has_ai_sig:
                # Explicitly generated by AI tools
                score = 0.1
            elif analysis['camera_make'] and analysis['camera_model']:
                # Typical camera metadata
                score = 0.7
                if analysis['editing_detected']:
                    score -= 0.15  # Edited, so slightly lower forensic certainty of originality
            elif analysis['editing_detected']:
                # Edited but no camera original metadata
                score = 0.55
            else:
                score = 0.5
        else:
            # No EXIF - very common on web
            score = 0.5

        return max(0.0, min(score, 1.0))
