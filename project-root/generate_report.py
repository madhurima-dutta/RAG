import pdfkit
from jinja2 import Environment, FileSystemLoader
import os

# Your data (same as before)
data = {
    "submission_id": "trn:oid:::13909:100400776",
    "submission_date": "Jun 11, 2025, 10:22 PM GMT+5:30",
    "download_date": "Jun 11, 2025, 10:25 PM GMT+5:30",
    "file_name": "final year 8th sem project.docx",
    "file_size": "652.3 KB",
    "page_count": "62 Pages",
    "word_count": "9,750 Words",
    "char_count": "57,452 Characters",
    "context_similarity": "85%",
    "citation_file": "final year 8th sem project.docx",
    "citation_page": "12",
    "citation_paragraph": "3",
    "entities": [
        {"type": "Organization", "value": "OpenAI"},
        {"type": "Concept", "value": "artificial general intelligence"},
        {"type": "Quantity", "value": "all of humanity"}
    ],
    "sentiment": "Neutral",
    "tone": "Formal",
    "summary": "This document provides an overview of OpenAI's mission, its approach to AGI safety, and strategic plans to ensure AI benefits humanity.",
    "qa_pairs": [
        {"q": "What is the mission of OpenAI?", "a": "OpenAI’s mission is to ensure that artificial general intelligence benefits all of humanity."},
        {"q": "How does OpenAI plan to ensure safety?", "a": "OpenAI employs rigorous safety research and policy development to align AGI with human values."}
    ],
    "page_number": 1,
    "total_pages": 3  # Example static page count, can make dynamic later
}

# Setup Jinja2 template loader
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('report_template.html')
html_out = template.render(**data)

# Save rendered HTML to file for debug/preview
with open('report_preview.html', 'w', encoding='utf-8') as f:
    f.write(html_out)

# Define wkhtmltopdf configuration
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

# PDF options for better control (disable network, load local files, etc.)
options = {
    'enable-local-file-access': '',
    'quiet': '',
    'disable-external-links': '',
    'disable-javascript': '',
    'encoding': 'UTF-8',
    'margin-top': '10mm',
    'margin-bottom': '10mm',
    'margin-left': '15mm',
    'margin-right': '15mm'
}

# Generate the PDF
try:
    pdfkit.from_file('report_preview.html', 'report_output.pdf', configuration=config, options=options)
    print("✅ PDF generated successfully: report_output.pdf")
except OSError as e:
    print("❌ PDF generation failed!")
    print(str(e))
