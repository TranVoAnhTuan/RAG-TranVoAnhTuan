link = "/home/jacktran/Downloads/draft/qnu.pdf"
import camelot

tables = camelot.read_pdf(
    link,
    pages="1",
    flavor="lattice"
)
df = tables[0].df
print(df)

table_json = tables[0].df.to_dict(orient="records")
print(table_json)




# from pdfminer.high_level import extract_text

# text = extract_text(link, page_numbers=[0])
# print(text)





# from io import StringIO

# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfdocument import PDFDocument
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfparser import PDFParser

# output_string = StringIO()
# with open('/home/jacktran/Downloads/draft/finance.pdf', 'rb') as in_file:
#     parser = PDFParser(in_file)
#     doc = PDFDocument(parser)
#     rsrcmgr = PDFResourceManager()
#     device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     for page in PDFPage.create_pages(doc):
#         interpreter.process_page(page)

# print(output_string.getvalue())

# from pdfminer.high_level import extract_pages
# from pdfminer.layout import LTTextContainer

# rows = []
# for page in extract_pages(link, page_numbers=[16]):
#     for element in page:
#         if isinstance(element, LTTextContainer):
#             x0, y0, x1, y1 = element.bbox
#             text = element.get_text().strip()
#             rows.append((y0, x0, text))

# rows.sort(reverse=True)  # từ trên xuống

# table = []
# current_row = []
# current_y = None

# for y, x, text in rows:
#     if current_y is None or abs(y - current_y) < 5:
#         current_row.append((x, text))
#     else:
#         table.append(sorted(current_row))
#         current_row = [(x, text)]
#     current_y = y

# table.append(sorted(current_row))
# print(table)