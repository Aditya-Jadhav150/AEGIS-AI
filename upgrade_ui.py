import re

def upgrade_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Inject CSRF meta
    if 'csrf-token' not in content:
        content = content.replace('<head>', '<head>\n    <meta name="csrf-token" content="{{ csrf_token() }}">')

    # 2. Inject CSRF into fetches
    fetch_pattern = re.compile(r"(fetch\([^,]+,\s*\{\s*method:\s*['\"]POST['\"],\s*headers:\s*\{)([^}]+)(\})")
    def fetch_repl(m):
        headers_inner = m.group(2)
        if 'X-CSRFToken' not in headers_inner:
            if headers_inner.strip():
                headers_inner += ", 'X-CSRFToken': document.querySelector('meta[name=\"csrf-token\"]').content"
            else:
                headers_inner = "'X-CSRFToken': document.querySelector('meta[name=\"csrf-token\"]').content"
        return f"{m.group(1)}{headers_inner}{m.group(3)}"
    
    content = fetch_pattern.sub(fetch_repl, content)

    # 3. Enhance CSS variables for WOW factor
    content = re.sub(r'--bg-dark:\s*#[0-9a-fA-F]+;', '--bg-dark: #050509;', content)
    content = re.sub(r'--bg-panel:\s*#[0-9a-fA-F]+;', '--bg-panel: rgba(18, 18, 24, 0.65);', content)
    content = re.sub(r'--bg-glass:\s*rgba\([^)]+\);', '--bg-glass: rgba(18, 18, 24, 0.85);', content)
    content = re.sub(r'--border-glow:\s*rgba\([^)]+\);', '--border-glow: rgba(0, 240, 255, 0.15);', content)
    content = re.sub(r'--border-active:\s*#[0-9a-fA-F]+;', '--border-active: #00F0FF;', content)
    content = re.sub(r'--accent:\s*#[0-9a-fA-F]+;', '--accent: #00F0FF;', content)
    content = re.sub(r'--accent-glow:\s*rgba\([^)]+\);', '--accent-glow: rgba(0, 240, 255, 0.4);', content)
    content = re.sub(r'--danger:\s*#[0-9a-fA-F]+;', '--danger: #FF2A55;', content)
    content = re.sub(r'--danger-glow:\s*rgba\([^)]+\);', '--danger-glow: rgba(255, 42, 85, 0.4);', content)

    # Add backdrop filter to body/panels
    content = content.replace('background-color: var(--bg-dark);', 'background-color: var(--bg-dark);\n            background-image: radial-gradient(circle at 15% 50%, rgba(0, 240, 255, 0.05), transparent 25%), radial-gradient(circle at 85% 30%, rgba(0, 255, 136, 0.05), transparent 25%);\n            background-attachment: fixed;')
    content = content.replace('background: var(--bg-glass);', 'background: var(--bg-glass);\n            backdrop-filter: blur(16px);\n            -webkit-backdrop-filter: blur(16px);')
    content = content.replace('.btn:hover {', '.btn:hover {\n            box-shadow: 0 0 15px var(--accent-glow);\n            transform: scale(1.02);')
    content = content.replace('.btn-primary {', '.btn-primary {\n            box-shadow: 0 4px 15px var(--accent-glow);')

    # 4. If index.html, fix Video
    if 'index.html' in filepath:
        # Fix accept
        content = content.replace('accept="image/*"', 'accept="image/*,video/*"')
        
        # Inject Video DOM
        img_orig_tag = '<img id="result-img-orig" src="" alt="Uploaded original">'
        new_dom = '<img id="result-img-orig" src="" alt="Uploaded original" style="display: none;">\n                            <video id="result-video-orig" src="" autoplay loop muted playsinline style="display: none; width: 100%; border-radius: 8px; border: 1px solid var(--border-glow); box-shadow: 0 0 20px var(--accent-glow);"></video>'
        content = content.replace(img_orig_tag, new_dom)
        
        img_align_tag = '<img id="result-img-aligned" src="" alt="Aligned Face">'
        content = content.replace(img_align_tag, '<img id="result-img-aligned" src="" alt="Aligned Face" style="display: none;">')
        
        # Inject Video JS Logic
        js_orig = "const resultImgOrig = document.getElementById('result-img-orig');"
        content = content.replace(js_orig, js_orig + "\n        const resultVideoOrig = document.getElementById('result-video-orig');")
        
        populate_orig = """            // Populate images
            resultImgOrig.src = data.image_url;
            resultImgAligned.src = data.aligned_url || data.image_url;"""
        
        populate_new = """            // Populate media
            if (data.is_video) {
                resultImgOrig.style.display = 'none';
                resultVideoOrig.style.display = 'block';
                resultVideoOrig.src = data.image_url;
                resultImgAligned.style.display = 'none';
            } else {
                resultVideoOrig.style.display = 'none';
                resultImgOrig.style.display = 'block';
                resultImgOrig.src = data.image_url;
                resultImgAligned.style.display = 'block';
                resultImgAligned.src = data.aligned_url || data.image_url;
            }"""
        content = content.replace(populate_orig, populate_new)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

upgrade_html('templates/index.html')
upgrade_html('templates/login.html')
upgrade_html('templates/admin.html')
