from fpdf import FPDF

def test():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    
    txt = "📌 Análise focada na hora prevista: 08:00"
    txt = txt.encode('latin-1', 'ignore').decode('latin-1')
    
    print(repr(txt))
    pdf.multi_cell(0, 6, txt)
    pdf.output("test.pdf")

test()
