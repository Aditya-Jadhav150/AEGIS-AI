from markdown_pdf import MarkdownPdf, Section
import os

try:
    pdf = MarkdownPdf(toc_level=0)
    with open('d:/ANTIGRAVITY/Work_file.md', 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split('<div style="page-break-before: always;"></div>')
    for part in parts:
        pdf.add_section(Section(part))

    pdf.save('d:/ANTIGRAVITY/Work_file.pdf')
    print("Successfully generated Work_file.pdf")
except Exception as e:
    print(f"Failed to generate: {e}")
