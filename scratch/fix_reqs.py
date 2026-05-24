import os
with open('d:/ANTIGRAVITY/requirements.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
lines = [l for l in lines if 'insightface' not in l and 'diffusers' not in l]
with open('d:/ANTIGRAVITY/requirements.txt', 'w', encoding='utf-8') as f:
    for l in lines:
        if l.strip() != "":
            f.write(l.strip() + '\n')
    f.write('insightface\n')
    f.write('diffusers\n')
