from transformers import pipeline

pipe = pipeline(
    task = 'text-classification',
    model = 'cross-encoder/qnli-electra-base'
)

# QNLI example
print(pipe("Where is Bucharest located?, Bucharest is in France."))

'''

classifier = pipeline(
    task = 'zero-shot-classification',
    model = 'facebook/bart-large-mnli'
)

text = "Hey, do you want to participate in our next add?"
categories = ['IT', 'marketing', 'support', 'finance']

output = classifier(text, categories)

print(f'Top label is: {output['labels'][0]} with score {output['scores'][0]}')

'''

# Abstractive vs Extractive summarization

summarizer = pipeline(
    task = 'summarization',
    model = 'Falconsai/text_summarization'
)

text = "Amy Jade Winehouse was an English singer, songwriter, musician, and businesswoman. She was known for her distinctive contralto vocals, expressive and autobiographical songwriting, and eclectic blend of genres such as soul, rhythm and blues, and jazz A cultural icon of the 21st century, Winehouse sold over 30 million records worldwide and won six Grammy Awards among other accolades. Winehouse was a member of the National Youth Jazz Orchestra in her youth, signing to Simon Fuller's 19 Management in 2002 and soon recording a number of songs before signing a publishing deal with EMI. She also formed a working relationship with producer Salaam Remi through these record publishers. Winehouse's debut album, Frank, was released in 2003. Many of the album's songs were influenced by jazz and, apart from two covers, were co-written by Winehouse. Frank was a critical and commercial success in the UK, and beyond, and was nominated for the UK's Mercury Prize. The song Stronger Than Me won her the Ivor Novello Award for Best Contemporary Song from the British Academy of Songwriters, Composers and Authors."

summary_text = summarizer(text)

print(summary_text[0]['summary_text'])