from flask import Flask, render_template, request
import nltk
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import pandas as pd
import speech_recognition as sr
import scipy
from summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
app = Flask(__name__)
nltk.download('punkt')

# Define a function to fetch transcript text from YouTube video URL or audio file
def get_transcript(source):
    if "youtube.com" in source or "youtu.be" in source:
        if "watch?v=" in source:
            video_id = source.split("watch?v=")[1].split("&")[0]
        elif "youtu.be" in source:
            video_id = source.split("youtu.be/")[1].split("?")[0]
        else:
            return "Invalid YouTube URL format."
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([line['text'] for line in transcript])
        except TranscriptsDisabled:
            return "Transcripts are disabled for this video."
        except Exception as e:
            return "An error occurred: " + str(e)
    else:
        r = sr.Recognizer()
        with sr.AudioFile(source) as source_audio:
            audio_text = r.listen(source_audio)
            try:
                transcript_text = r.recognize_google(audio_text)
            except Exception as e:
                return "An error occurred: " + str(e)
    return transcript_text

def summarize_text(transcript_text):
    summarizer = pipeline('summarization')
    original_length = len(transcript_text.split())
    # Split the text into chunks of 1024 words
    chunks = [transcript_text[i:i+1024] for i in range(0, len(transcript_text), 1024)]
    summarized = []
    for chunk in chunks:
        # Summarize each chunk
        chunk_summary = summarizer(chunk, min_length=1, do_sample=False)
        if chunk_summary and 'summary_text' in chunk_summary[0]:
            summarized.append(chunk_summary[0]['summary_text'])
    if not summarized:
        return "No summary could be generated.", "Summarization done: 0%"
    summary_text = ' '.join(summarized)
    summary_length = len(summary_text.split())
    summarization_ratio = round((summary_length / original_length) * 100, 2)

    return summary_text, f"Summarization done: {summarization_ratio}%"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the YouTube video URL from the form
        url = request.form.get('url')

        # Get the transcript of the YouTube video
        transcript_text = get_transcript(url)

        if not transcript_text:
            transcript = "No transcript available for this video."
            summarized_text = None
        else:
            transcript = transcript_text
            # Summarize the transcript
            summarized_text = summarize_text(transcript_text)
            if not summarized_text:
                summarized_text = "No summary could be generated for this transcript."

        # Pass the transcript and summarized text to the template
        return render_template('index.html', transcript=transcript, summary=summarized_text)

    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def transcript():
    source = request.form['source']

    # Get the transcript of the YouTube video
    transcript_text = get_transcript(source)

    if not transcript_text:
        transcript = "No transcript available for this video."
        summarized_text = None
    else:
        transcript = transcript_text
        # Summarize the transcript
        summarized_text = summarize_text(transcript_text)
        # print(f"Summarized text: {summarized_text}")

    # Pass the transcript and summarized text to the template
    return render_template('summary.html', transcript=transcript, summary=summarized_text)

if __name__ == '__main__':
    app.run(debug=True)