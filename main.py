from flask import Flask, render_template
import os
import random
import json
import pandas as pd
import torch
import sys
from sentence_transformers import SentenceTransformer, util
import collections
import subprocess
from transformers import T5ForConditionalGeneration, T5Tokenizer

APP = Flask(__name__)

use_whisper = False
whisper_location = "" # Location of the streaming capable whisper binary (see: https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream)
whisper_model_location = "" # location of the model file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_quotes_file():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(current_dir, 'quotes.parquet')):
        print("Downloading quotes.parquet...")
        # download from huggingface
        url = "https://huggingface.co/datasets/gobbledegook/quote-embeddings/resolve/main/quotes.parquet"
        # download to current directory with curl. make sure to save to current directory
        os.system(f"curl -L {url} -o {current_dir}/quotes.parquet")
    else:
        print("Quotes file found")

SEMANTIC_SEARCH = None
def get_semantic_engine():
    """Get the embedder"""
    print("Loading embedding file...")
    download_quotes_file()
    quote_file = pd.read_parquet("quotes.parquet") # swap to quotes-small.parquet for a smaller dataset
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Converting to tensor...")
    corpus = quote_file['quote'].tolist()
    corpus_embeddings = torch.tensor(
        quote_file['embeddings'].tolist()).float().to(device)
    top_k = min(6, len(corpus))
    return collections.namedtuple('Engine', ['embedder', 'corpus', 'corpus_embeddings', 'top_k'])(embedder, corpus, corpus_embeddings, top_k)
def get_semantic_suggestions(prompt):
    global SEMANTIC_SEARCH
    if not SEMANTIC_SEARCH:
        print("Loading semantic suggestions...")
        SEMANTIC_SEARCH = get_semantic_engine()
    query_embedding = SEMANTIC_SEARCH.embedder.encode(prompt, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, SEMANTIC_SEARCH.corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=SEMANTIC_SEARCH.top_k)
    final = []
    for _, idx in zip(top_results[0], top_results[1]):
            final.append({'text': SEMANTIC_SEARCH.corpus[idx]})
    return final 


MODEL = None
TOKENIZER = None


def get_llm_suggestions(prompt):
    """Call ./llm and return the output"""
    global MODEL, TOKENIZER
    if MODEL is None:
        print("Loading model...")
        model_name = "gobbledegook/t5-small-lm-adapt-quotes"
        MODEL = T5ForConditionalGeneration.from_pretrained(model_name)
        TOKENIZER = T5Tokenizer.from_pretrained(model_name)
    prompt = "write: " + prompt
    input_ids = TOKENIZER.encode(prompt, return_tensors="pt")
    outputs = MODEL.generate(input_ids, max_length=100, temperature=0.7, num_beams=6, num_return_sequences=6)
    final = []
    for output in outputs:
        final.append({'text': TOKENIZER.decode(output, skip_special_tokens=True)})
    return final


PROCESS = None
def get_hear():
    global PROCESS
    """Call ./hear and return the output"""
    if PROCESS is None or PROCESS.poll() is not None:
        if use_whisper:
            print("Using whisper for transcription")
            PROCESS = subprocess.Popen(
                [whisper_location, '-f', '/tmp/whisper.out', '-m',
                 whisper_model_location])
            # create file
            with open('/tmp/whisper.out', 'w') as f:
                pass
            return
        else:
            print("Starting ./hear")
            # call ./hear in a subprocess as to not block the main thread
            PROCESS = subprocess.Popen(
                ["./hear"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    else:
        print("Using existing ./hear process")
    return PROCESS

def random_words():
    """Mock words generator"""
    random_words = ['hello', 'world', 'foo', 'bar', 'baz', 'qux', 'quux',
                    'corge', 'grault', 'garply', 'waldo', 'fred', 'plugh', 'xyzzy', 'thud']
    return ' '.join(random.sample(random_words, 10))

@APP.route('/')
def index():
    return render_template('index.html')

@APP.route('/stream')
def speech():
    # return a long response
    def generate():
        # get the next few words from the generator
        last_line = None
        while True:
            word_source = None
            hear_process = get_hear()
            if use_whisper:
                with open('/tmp/whisper.out', 'r') as f:
                    words = f.read().strip()
                    # split on newlines
                    words = words.split('\n')
                    word_source = words
                    if len(words) > 0:
                        word_source = [words[-1]]
                    if "BLANK_AUDIO" in words:
                        continue
            else:
                word_source = hear_process.stdout
            for line in word_source:
                if use_whisper:
                    words = line
                    if len(words) == 0:
                        continue
                else:
                    words = line.decode('utf-8').strip()
                if words == last_line:
                    continue
                if words[-1] == '.':
                    # reset
                    words = ''
                # if there is a period, we have a sentence, get everything after first period
                elif '.' in words:
                    words = words.split('.')[-1]
                last_line = words
                # only keep last 20 words
                words = ' '.join(words.split()[-20:])
                output = {"transcript": words, "suggestions": get_suggestions(words)}
                output = json.dumps(output)
                yield 'data: {}\n\n'.format(output)
    return APP.response_class(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    if 'llm' in sys.argv:
        print("Using LLM for suggestions")
        get_suggestions = get_llm_suggestions
    else:
        print("Using semantic search for suggestions")
        get_suggestions = get_semantic_suggestions
    print("Starting up...")
    get_suggestions("test")
    APP.run(debug=False, port=5001, threaded=True)