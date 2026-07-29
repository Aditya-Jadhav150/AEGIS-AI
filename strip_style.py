def strip_style(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start, end = -1, -1
    for i, line in enumerate(lines):
        if "<style>" in line and start == -1:
            start = i
        if "</style>" in line and start != -1:
            end = i
            break
    if start != -1 and end != -1:
        lines[start] = "    <link rel=\"stylesheet\" href=\"{{ url_for('static', filename='css/main.css') }}\">\n"
        del lines[start+1:end+1]
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
strip_style("templates/login.html")
strip_style("templates/admin.html")
