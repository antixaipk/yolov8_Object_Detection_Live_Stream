from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Define the text to be summarized
text = "Talks between the assistant ministers of foreign affairs of Russia, Iran,Syria and the Turkey began at the headquarters of the Russian Ministry of Foreign Affairs in Moscow."
# Parse the text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Summarize the text using Latent Semantic Analysis (LSA)
summarizer = LsaSummarizer()
summary = summarizer(parser.document, sentences_count=1)

# Print the summary
print(summary)
