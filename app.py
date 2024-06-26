import logging
import streamlit as st
import requests
from xml.etree import ElementTree
from openai import OpenAI, AssistantEventHandler
import os
from streamlit_lottie import st_lottie_spinner
import json
from typing_extensions import override
from io import StringIO
from datetime import datetime
import pandas as pd
from pdfreader import SimplePDFViewer, PageDoesNotExist
import io
import docx

OPENAI_API_KEY = st.secrets["openai"]["api_key"]
ASSISTANT_ID = st.secrets["assistant"]["id"]
CORE_API_KEY = st.secrets["core"]["api_key"]
CORE_BASE_URL = st.secrets["core"]["base_url"]

assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"
assert CORE_API_KEY, "CORE_API_KEY is not set"

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log_demo.txt"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

logger.info("Environment variables loaded successfully")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized")

class EventHandler(AssistantEventHandler):
    def __init__(self, placeholder=None):
        super().__init__()
        self.text_accumulated = ''
        self.placeholder = placeholder
        self.first_chunk = True  # To handle the first chunk correctly
        logger.info("EventHandler initialized")

    @override
    def on_text_created(self, text):
        # Do not immediately set text_accumulated to text.value to avoid duplication
        logger.info(f"Text created: {text.value}")

    @override
    def on_text_delta(self, delta, snapshot):
        if self.first_chunk:
            self.text_accumulated = delta.value
            self.first_chunk = False
        else:
            self.text_accumulated += delta.value
        if self.placeholder:
            self.placeholder.markdown(self.text_accumulated)

    @override
    def on_text_done(self, text):
        # No need to display the text again
        logger.info(f"Text done: {text.value}")

def create_assistant():
    try:
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="Always answer the user's questions accurately, with the appropriate context",
            model="gpt-4o"
        )
        logger.info(f"Assistant created: {assistant.id}")
        return assistant
    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        raise

if 'assistant' not in st.session_state:
    st.session_state.assistant = create_assistant()

def run_assistant(thread_id, assistant_id, task, placeholder=None):
    handler = EventHandler(placeholder)
    logger.info(f"Running assistant {assistant_id} for thread {thread_id} with task: {task}")
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=handler,
        model="gpt-4o",
        temperature=0.3,
    ) as stream:
        stream.until_done()
    logger.info(f"Assistant run completed. Accumulated text: {handler.text_accumulated[:50]}...")  # Displaying only the first 50 characters
    return handler.text_accumulated

def create_thread():
    thread = client.beta.threads.create()
    logger.info(f"Thread created: {thread.id}")
    return thread

def add_message_to_thread(thread_id, content, role="user"):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content
    )
    logger.info(f"Message added to thread {thread_id}: {content[:50]}...")  # Displaying only the first 50 characters
    return message

# Function to load Lottie files
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            logger.info(f"Loading Lottie file from {filepath}")
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File {filepath} not found.")
        logger.error(f"File {filepath} not found.")
        return None

# Load Lottie animation for loading
loading_animation = load_lottiefile("animation.json")

# Custom CSS for fonts and styles
st.markdown("""
    <style>
        @media screen and (max-width: 600px) {
            .main-title {
                font-size: 48px;
            }
        }
        @media screen and (min-width: 601px) {
            .main-title {
                font-size: 60px;
            }
        }
        .main-title {
            font-family: 'Courier New', monospace;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
            background: linear-gradient(to right, blue, pink);
            -webkit-background-clip: text;
            color: transparent;
            margin-top: 40px;
        }
        .disclaimer {
            font-size: 14px;
            color: #888888;
            margin-bottom: 20px;
            text-align: center.
        }
        .input-area {
            font-family: 'Courier New', monospace;
            margin-top: 22px.
        }
        .custom-button {
            background-color: #4CAF50;
            color: white;
            border: none.
            padding: 10px 20px.
            font-size: 16px.
            cursor: pointer.
        }
        .custom-button:hover {
            background-color: #45a049.
        }
        .article-card {
            background-color: #f9f9f9.
            padding: 15px.
            border-radius: 10px.
            margin-bottom: 20px.
            border: 1px solid #ddd.
        }
        .article-title {
            font-size: 1.2em.
            color: #333.
            font-weight: bold.
        }
        .article-authors {
            color: #777.
        }
        .article-details {
            margin-top: 10px.
            font-size: 0.9em.
            color: #555.
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Brilliance (Project Hessdalen Beta)</h1>", unsafe_allow_html=True)

# Initialize conversation history and user_input in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    logger.info("Conversation history initialized")

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# User Input Section
st.markdown("<div class='input-area'><b></b></div>", unsafe_allow_html=True)
user_input = st.text_input("Research Question", key="user_input", placeholder="Enter your research question here...", label_visibility="collapsed")
generate = st.button("Generate", key='generate_button', help="Click to generate a response")
response_placeholder = st.empty()

# Setup assistant and thread outside the button click logic
if 'assistant' not in st.session_state:
    st.session_state.assistant = create_assistant()
if 'thread' not in st.session_state:
    st.session_state.thread = create_thread()

    
def optimize_question(thread_id, question):
    task = (
        f"Transform the question: {question} to be the very optimal, precise and clear. Only, I REPEAT: ONLY, output the optimized revised question."
    )
    add_message_to_thread(thread_id, task)
    response_text = run_assistant(thread_id, st.session_state.assistant.id, task)
    if response_text:
        optimized_question = response_text.strip()
        logger.info(f"Optimized question: {optimized_question}")
        return optimized_question
    else:
        logger.warning("No optimized question generated")
        return question

def extract_keywords(thread_id, question):
    optimized_question = optimize_question(thread_id, question)
    task = (
        f"Extract the most essential academic keywords from the following research question. Choose the most relevant 4 keywords max. "
        f"Research Question: {optimized_question}. Output keywords separated by commas, ranked from most relevant to least relevant."
    )
    add_message_to_thread(thread_id, task)
    response_text = run_assistant(thread_id, st.session_state.assistant.id, task)
    if response_text:
        keywords = response_text.split(',')
        clean_keywords = [keyword.strip() for keyword in keywords]
        logger.info(f"Extracted keywords: {clean_keywords}")
        return clean_keywords
    else:
        logger.warning("No keywords extracted")
        return []

# Add file uploader for users to upload their files
uploaded_files = st.file_uploader("Upload your files", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# Process uploaded files
uploaded_files_content = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the file name
        st.write(f"Uploaded file: {uploaded_file.name}")

        # Read the file content based on its type
        if uploaded_file.type == "text/plain":
            content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            uploaded_files_content.append(content)
            st.text_area(f"Content of {uploaded_file.name}", content)
        elif uploaded_file.type == "application/pdf":
            # Process PDF file
            with io.BytesIO(uploaded_file.getvalue()) as pdf_file:
                viewer = SimplePDFViewer(pdf_file)
                text = ""
                try:
                    while True:
                        viewer.render()
                        text += " ".join(viewer.canvas.strings)
                        viewer.next()
                except PageDoesNotExist:
                    pass
            uploaded_files_content.append(text)
            st.text_area(f"Content of {uploaded_file.name}", text)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Process Word document
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            text = "\n".join([para.text for para in doc.paragraphs])
            uploaded_files_content.append(text)
            st.text_area(f"Content of {uploaded_file.name}", text)         

def generate_response(thread_id, assistant_id, context, uploaded_content, placeholder):
    combined_context = context + "\n\n" + "\n\n".join(uploaded_content)
    prompt = (
        f"You are Brilliance, a GPT-4o model with access to major journals. Your sole task is to accurately, and I repeat: accurately, answer the user's question with empirical data. You will emulate a wide beam search when considering your choice of words. This is the most important thing to remember. Deliver a brilliant and detailed, scientifically validated response to {st.session_state.user_input}, using the context below, citing the studies' authors and years immediately after presenting facts. "
        f"Clarify the mechanisms of action when discussing medicine. Place a higher emphasis on the most recent studies but consider ALL relevant research. Craft the answer in natural, flowing language, avoiding numbered lists or subtopic breakdowns. Synthesize information from recent and cutting-edge research, "
        f"emphasizing groundbreaking discoveries and their practical implications. Highlight innovative theories or advancements that could revolutionize our understanding, "
        f"focusing on the unique aspects of the research question within the latest context. Reference the original question frequently, aiming for approximately 600 words. "
        f"Include accurate data, values, variables, and relevant names or places. Be specific, avoid generalizations, and eschew repetitive phrasing. Aim to leave the reader "
        f"with a profound understanding, using a natural academic tone suitable for an audience with an IQ of 200. Extrapolate and synthesize groundbreaking insights. "
        f"Ensure the question is completely and accurately answered, considering the data from \n\n{combined_context}\n\n. Make sure your results show groundbreaking findings. Remember to synthesize responses with citations in parentheses. Just use relevant author names and year in the prompt. "
        f"Shoot for 1000 words. Place high emphasis on cutting edge research, innovative applications and potential future directions that could revolutionize the field, all while accurately and PRECISELY, with clarity, i repeat: PRECISELY, with clarity, answering the user's question. Remember, your goal is definitively answer the user's question even if the specific synthesis of knowledge was not explicitly stated. Do not include a references section at the end, that is a separate feature. Make sure language is refined to emphasize key points and clarity. Question: {st.session_state.user_input}."
    )
    logger.info(f"Prompt: {prompt}")  # Print the prompt to see the user input with context and question
    add_message_to_thread(thread_id, prompt)
    response_text = run_assistant(thread_id, assistant_id, prompt, placeholder)
    logger.info(f"Generated response: {response_text[:50]}...")  # Displaying only the first 50 characters
    return response_text

def parse_publication_year(date_string):
    try:
        # Assuming date_string is in ISO format (e.g., "2023-05-05T15:35:30Z")
        date_obj = datetime.fromisoformat(date_string.rstrip("Z"))
        return date_obj.year
    except ValueError:
        logger.error(f"Error parsing date: {date_string}")
        return "Unknown Year"
    
def search_coreapi(keywords):
    if not keywords:
        logger.warning("No keywords provided for CORE API search.")
        return []

    # Limit the number of keywords to a manageable number (e.g., 3-5)
    limited_keywords = keywords[:5]
    query = ' OR '.join(limited_keywords)  # Use logical OR to combine keywords
    url = f"{CORE_BASE_URL}/search/works/"
    headers = {
        'Authorization': f'Bearer {CORE_API_KEY}'
    }
    params = {
        'q': query,
        'limit': 12
    }

    # Log the full API query details
    logger.info(f"CORE API URL: {url}")
    logger.info(f"CORE API Headers: {headers}")
    logger.info(f"CORE API Params: {params}")
    response = requests.get(url, headers=headers, params=params)
    logger.info(f"CORE API response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"Error fetching data from CORE API: {response.text}")
        return []

    data = response.json()
    logger.debug(f"CORE API response data: {json.dumps(data, indent=2)}")
    articles = []

    for result in data.get('results', []):
        authors = [author['name'] for author in result.get('authors', []) if 'name' in author]
        articles.append({
            'title': result.get('title', 'No title'),
            'abstract': result.get('abstract', 'No abstract'),
            'published': parse_publication_year(result.get('publishedDate', '')),
            'authors': authors,
            'source': 'CORE',
            'url': result.get('downloadUrl', '')
        })

    logger.info(f"CORE API search results: {len(articles)} articles found")
    return articles
    
def search_arxiv(keywords):
    if not keywords:
        logger.warning("No keywords provided for Arxiv search.")
        return []
    query = '+'.join(keywords)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=1&max_results=15"
    response = requests.get(url)
    logger.info(f"Arxiv API response status: {response.status_code}")
    root = ElementTree.fromstring(response.content)
    articles = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
        published_date = parse_publication_year(entry.find('{http://www.w3.org/2005/Atom}published').text)
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        articles.append({
            'title': title,
            'abstract': abstract,
            'published': published_date,
            'authors': authors,
            'source': 'Arxiv',
            'url': entry.find('{http://www.w3.org/2005/Atom}id').text
        })
    logger.info(f"Arxiv search results: {len(articles)} articles found")
    return articles

def search_plos(keywords):
    def get_plos_results(query):
        url = f"https://api.plos.org/search?q=everything:{query}&fl=id,title_display,abstract,author_display,journal,publication_date&wt=json&rows=7"
        try:
            response = requests.get(url, allow_redirects=True)
            logger.info(f"PLOS API response status: {response.status_code}")
            
            # Handle redirects
            if response.status_code == 308:
                new_url = response.headers['Location']
                logger.info(f"Following redirect to: {new_url}")
                response = requests.get(new_url)
                logger.info(f"Redirected PLOS API response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Error fetching data from PLOS: {response.text}")
                return []
            
            data = response.json()
            logger.debug(f"PLOS API response data: {json.dumps(data, indent=2)}")
            
            if 'response' not in data or 'docs' not in data['response']:
                logger.warning("No documents found in the PLOS response.")
                return []
            
            articles = []
            for doc in data['response']['docs']:
                articles.append({
                    'title': doc.get('title_display', 'No title'),
                    'abstract': ' '.join(doc.get('abstract', 'No abstract')),
                    'id': doc.get('id', ''),
                    'published': doc.get('publication_date', '')[:4],  # Taking only the year part
                    'authors': doc.get('author_display', []),
                    'source': doc.get('journal', 'PLOS'),
                    'url': f"https://journals.plos.org/plosone/article?id={doc.get('id', '')}"
                })
            logger.info(f"PLOS search results: {len(articles)} articles found")
            return articles
        except Exception as e:
            logger.error(f"Exception occurred while fetching data from PLOS: {e}")
            return []

    if not keywords:
        logger.warning("No keywords provided for PLOS search.")
        return []

    # First attempt with all keywords
    query = '+'.join(keywords)
    articles = get_plos_results(query)

    # If no results, retry with only the top 2 keywords
    if not articles:
        logger.info("No results with all keywords, retrying with top 2 keywords")
        query = '+'.join(keywords[:2])
        articles = get_plos_results(query)
        
    return articles


def display_article_card(article, is_dark_mode=False):
    # Ensure that the abstract is a string
    abstract = article.get('abstract', 'No abstract')
    if not isinstance(abstract, str):
        abstract = 'No abstract'
    
    # Split the abstract into lines and take the first 3
    abstract_lines = abstract.split('\n')
    first_3_lines = '\n'.join(abstract_lines[:3])

    st.markdown(f"""
    <div class='article-card'>
        <a href="{article['url']}" class='article-title'>{article['title']}</a>
        <div class='article-authors'>Authors: {', '.join(article['authors'])}</div>
        <div class='article-details'>
            <p>Published: {article['published']}</p>
            <p>Source: {article['source']}</p>
            <p>{first_3_lines}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    logger.info(f"Displayed article: {article['title']}")
    
# API selection checkboxes
use_arxiv = st.checkbox("Use Arxiv", value=True)
use_plos = st.checkbox("Use PLOS", value=True)
use_core = st.checkbox("Use CORE", value=True)

if generate:
    if user_input == "":
        st.warning("Please enter a message ⚠️")
    else:
        with st_lottie_spinner(loading_animation):
            logger.info(f"User input: {user_input}")

            # Extract keywords
            keywords = extract_keywords(st.session_state.thread.id, user_input)
            logger.info(f"Keywords: {keywords}")

            # Search databases based on user selection
            all_articles = []

            # Search Arxiv
            if use_arxiv:
                arxiv_articles = search_arxiv(keywords)
                all_articles.extend(arxiv_articles)

            # Search PLOS
            if use_plos:
                plos_articles = search_plos(keywords)
                all_articles.extend(plos_articles)

            # Search CORE API
            if use_core:
                core_articles = search_coreapi(keywords)
                all_articles.extend(core_articles)

            # Create context for response generation
            context = "\n".join(
                f"Title: {article['title']}\nAuthors: {', '.join(article['authors'])}\nPublished: {article['published']}\nAbstract: {article['abstract']}\n"
                for article in all_articles
                if article['title'] and article['abstract'] and article['authors']
            )
            logger.info(f"Context for response generation: {context[:50]}...") # Displaying only the first 50 characters

            # Generate response
            response = generate_response(st.session_state.thread.id, st.session_state.assistant.id, context, uploaded_files_content, response_placeholder)
            logger.info(f"Response: {response[:50]}...") # Displaying only the first 50 characters

            # Update conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            # Display the articles
            for article in all_articles:
                display_article_card(article, is_dark_mode=False)
                logger.info(f"Article displayed: {article['title']}")
