from pypdf import PdfReader
from transformers import pipeline

reader = PdfReader("../hggin/pdf/Alexandru_Stoica.pdf")

document_text = ''
for page in reader.pages:
    document_text += page.extract_text()

#print(document_text[:200])

my_pipe = pipeline(
    task = 'question-answering',
    model = 'distilbert-base-cased-distilled-squad'
)

question = 'Where did Alexandru Stoica study?'
result = my_pipe(question = question, context = document_text)
print(f'The answer is: {result['answer']}')
