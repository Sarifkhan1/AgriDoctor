import os
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_document():
    doc = Document()
    
    # Colors
    COLOR_PRIMARY = RGBColor(46, 117, 89)      # Forest Green
    COLOR_SECONDARY = RGBColor(33, 37, 41)    # Off Black
    COLOR_MUTED = RGBColor(108, 117, 125)     # Gray
    
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Arial'
    font.size = Pt(11)
    font.color.rgb = COLOR_SECONDARY
    
    # Title Page
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("\n\nAGRIDOCTOR AI\n")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(28)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY

    subtitle_run = title_p.add_run("A Safety-Constrained Vision-Language and Multimodal Framework for Mobile Crop-Leaf Pathology and Livestock Health Diagnostics\n\n\n\n")
    subtitle_run.font.name = 'Arial'
    subtitle_run.font.size = Pt(14)
    subtitle_run.font.color.rgb = COLOR_MUTED

    details_p = doc.add_paragraph()
    details_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    details_p.paragraph_format.line_spacing = 1.3
    
    details_text = (
        "Module: AI-4-Creativity Project (Final Major Project)\n"
        "Student Name: Mohammad Sarif Khan\n"
        "Student ID: 2238572\n"
        "Submission Category: Resit Submission Report\n"
        "Evaluation Note: Capped at 40% Record / Evaluated at Full Quality Standard\n\n"
        "Academic Year: 2026\n"
    )
    run_details = details_p.add_run(details_text)
    run_details.font.size = Pt(11)
    run_details.font.color.rgb = COLOR_SECONDARY
    
    doc.add_page_break()
    
    # Read the text report file
    report_path = "/Users/santosh/Documents/agri-doctor/resit_submission_report.txt"
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return
        
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_code_block = False
    code_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for code block toggle
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Inches(0.4)
                p.paragraph_format.line_spacing = 1.0
                
                pPr = p._p.get_or_add_pPr()
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
                pPr.append(shading)
                
                run = p.add_run("\n".join(code_lines))
                run.font.name = 'Consolas'
                run.font.size = Pt(9.5)
                run.font.color.rgb = COLOR_SECONDARY
                
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line.rstrip("\n"))
            i += 1
            continue
            
        # Ignore raw divider lines in text file
        if stripped.startswith("===") or stripped.startswith("---"):
            i += 1
            continue
            
        # Check for Heading 1 (underlined by === or surrounded by ===)
        is_h1 = False
        if i + 1 < len(lines) and lines[i+1].strip().startswith("==="):
            is_h1 = True
            h1_text = stripped
            i += 2  # skip current and next divider line
        elif i - 1 >= 0 and lines[i-1].strip().startswith("===") and i + 1 < len(lines) and lines[i+1].strip().startswith("==="):
            # Already handled by the forward look
            i += 1
            continue
            
        if is_h1:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h1_text)
            run.font.name = 'Arial'
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = COLOR_PRIMARY
            continue
            
        # Check for Heading 2 (underlined by --- or matches sub-headers like 2.1)
        is_h2 = False
        if i + 1 < len(lines) and lines[i+1].strip().startswith("---"):
            is_h2 = True
            h2_text = stripped
            i += 2
        elif (stripped.startswith("2.") or stripped.startswith("3.") or stripped.startswith("4.") or stripped.startswith("5.") or stripped.startswith("6.") or stripped.startswith("7.") or stripped.startswith("8.")) and (" " in stripped) and len(stripped.split()[0]) <= 5:
            # Matches "2.1 Introduction"
            is_h2 = True
            h2_text = stripped
            i += 1
            
        if is_h2:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h2_text)
            run.font.name = 'Arial'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = COLOR_PRIMARY
            continue
            
        # Empty lines
        if not stripped:
            i += 1
            continue
            
        # Regular bullet lists
        if stripped.startswith("* ") or stripped.startswith("- "):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.line_spacing = 1.15
            
            bullet_text = stripped[2:]
            # Bold prefix if there is a colon (e.g. "* Bold: text")
            if ":" in bullet_text and len(bullet_text.split(":")[0]) < 40:
                parts = bullet_text.split(":", 1)
                r1 = p.add_run("• " + parts[0] + ":")
                r1.font.bold = True
                p.add_run(parts[1])
            else:
                p.add_run("• " + bullet_text)
            i += 1
            continue
            
        # Regular paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        
        # Bold prefix check for inline lists or key/value highlights (e.g. "Name: Value")
        if ":" in stripped and len(stripped.split(":")[0]) < 40 and not stripped.startswith("http"):
            parts = stripped.split(":", 1)
            r1 = p.add_run(parts[0] + ":")
            r1.font.bold = True
            p.add_run(parts[1])
        else:
            p.add_run(stripped)
        i += 1

    doc.save("/Users/santosh/Documents/agri-doctor/AgriDoctor_Resit_Submission_Report.docx")
    print("Success: Generated AgriDoctor_Resit_Submission_Report.docx dynamically.")

if __name__ == "__main__":
    create_document()
