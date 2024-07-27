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
from PIL import Image
import time

# Custom CSS for layout and spacing
st.markdown("""
    <style>
        .main-title {
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .input-area, .api-selection, .response-section {
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .custom-button {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for color scheme and typography
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            color: #4CAF50;
        }
        .input-area, .api-selection, .response-section {
            background-color: #ffffff;
            border: 1px solid #ddd;
        }
        .custom-button {
            background-color: #4CAF50;
            color: white;
        }
        .custom-button:hover {
            background-color: #45a049;
        }
        .upload-box {
            background-color: #e0e0e0;
            border: 2px dashed #ccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-box:hover {
            background-color: #d0d0d0;
        }
        .search-box {
            width: 100%;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-bottom: 20px;
        }
        .options-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1em;
            font-weight: normal;  /* Remove bold */
            color: #333;
        }
        .options-container label {
            font-size: 1em;
            color: #333;
        }
        .api-warning {
            display: none;
            color: #FFA500;
            font-size: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Load Google Fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


OPENAI_API_KEY = st.secrets["openai"]["api_key"]
NCBI_BASE_URL = st.secrets["ncbi"]["base_url"]
SPRINGER_API_KEY = st.secrets["springer"]["api_key"]
SPRINGER_BASE_URL = st.secrets["springer"]["base_url"]
ASSISTANT_ID = st.secrets["assistant"]["id"]
CORE_API_KEY = st.secrets["core"]["api_key"]
CORE_BASE_URL = st.secrets["core"]["base_url"]
CROSSREF_BASE_URL = "https://api.crossref.org"

assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"
assert NCBI_BASE_URL, "NCBI_BASE_URL is not set"
assert SPRINGER_API_KEY, "SPRINGER_API_KEY is not set"
assert SPRINGER_BASE_URL, "SPRINGER_BASE_URL is not set"
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

# Define API specialties with emojis
api_specialties = {
    "Arxiv": "Physics, Mathematics 🔬",
    "PLOS": "Biology, Medicine 🌿",
    "Springer": "Science, Medicine ⚙️",
    "NCBI": "Biology, Medicine 🧬",
    "CORE": "Research, Publications 📚",
    "Crossref": "Metadata, DOI, References 🔗"
}

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
    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a research assistant. Your beams are wide, however accuracy is of the utmost importance
        """,
        model="gpt-4o"
    )
    logger.info(f"Assistant created: {assistant.id}")
    return assistant
    
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
    # Retrieve all runs for the thread
    runs = client.beta.threads.runs.list(thread_id=thread_id)
    
    # Check if there are any active runs
    active_run = next((run for run in runs if run.status in ["queued", "in_progress"]), None)
    
    while active_run:
        logger.info(f"Waiting for active run to complete for thread {thread_id}")
        time.sleep(1)  # Wait for a second before checking again
        runs = client.beta.threads.runs.list(thread_id=thread_id)
        active_run = next((run for run in runs if run.status in ["queued", "in_progress"]), None)
    
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
            margin-top: 22px;
        }
        .custom-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .custom-button:hover {
            background-color: #45a049;
        }
        .article-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }
        .article-title {
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }
        .article-authors {
            color: #777;
        }
        .article-details {
            margin-top: 10px;
            font-size: 0.9em;
            color: #555;
        }
        .upload-box {
            background-color: #e0e0e0;
            border: 2px dashed #ccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-box:hover {
            background-color: #d0d0d0;
        }
        .search-box {
            width: 100%;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-bottom: 20px;
        }
        .options-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Brilliance</h1>", unsafe_allow_html=True)

# Upload Box
uploaded_files = st.file_uploader("", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# User Input Section
user_input = st.text_input("", key="user_input", placeholder="Enter your research question here...", label_visibility="collapsed", max_chars=200)
generate = st.button("Generate", key='generate_button', help="Click to generate a response", use_container_width=True)
response_placeholder = st.empty()

# Custom CSS for layout and spacing
st.markdown("""
    <style>
        .main-title {
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .input-area, .api-selection, .response-section {
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .custom-button {
            margin-top: 10px;
        }
        .options-container .section-title {
            font-size: 16px;
            font-weight: 400;
            color: red;  /* Change text color to red */
            margin-bottom: 10px;
            position: relative;
        }
        .options-container .section-title::after {
            content: "";
            display: block;
            width: 100%;
            border-bottom: 2px solid red;
            position: absolute;
            left: 0;
            bottom: -5px;
        }
        .options-container .section-title::before {
            content: "";
            display: block;
            width: 100%;
            border-bottom: 2px solid red;
            position: absolute;
            left: 0;
            bottom: -10px;
        }
        .api-warning {
            color: #ff4d4f;
            font-size: 14px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Options Section with Expanders
st.markdown("<div class='options-container'>", unsafe_allow_html=True)

with st.expander("Select response length:", expanded=False):
    length_selection = st.radio(
        "",
        ("Short", "Medium", "Full/Long"),
        index=1,  # Default to "Medium"
        key="length_radio"
    )

with st.expander("Sources:", expanded=False):
    all_selected = st.checkbox("All (Select all APIs)", key="checkbox_all")
    selected_apis = []
    for api, specialty in api_specialties.items():
        if st.checkbox(f"{api} ({specialty})", key=f"checkbox_{api}", value=all_selected):
            selected_apis.append(api)

    # Placeholder for the warning message
    api_warning = st.empty()

if not selected_apis:
    api_warning.markdown("<div class='api-warning'>Please select at least one API.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

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

# Initialize conversation history and user_input in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    logger.info("Conversation history initialized")
    
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

def generate_response(question, length, context, uploaded_content, placeholder):
    if not question.strip():
        st.error("Please enter a valid research question.")
        return ""
    
    combined_context = context + "\n\n" + "\n\n".join(uploaded_content)
    
    def generate_prompt(optimized_question, length):
        base_prompt = (
            f"You are Brilliance, a GPT-4o model. Answer '{optimized_question}' using context below:\n\n{combined_context}\n\n"
        )
        length_prompts = {
            "Short": f"""
            You are Brilliance, a GPT-4o model with access to major journals. Your sole task is to accurately, and I repeat: accurately, answer the user's question with empirical data. You will emulate a wide beam search when considering your choice of words. This is the most important thing to remember. Deliver a brilliant and detailed, scientifically validated response to {optimized_question}, using the context below, citing the studies' authors and years immediately after presenting facts. 
            Clarify the mechanisms of action when discussing medicine. Place a higher emphasis on the most recent studies but consider ALL relevant research. Craft the answer in natural, flowing language, avoiding numbered lists or subtopic breakdowns. Synthesize information from recent and cutting-edge research, 
            emphasizing groundbreaking discoveries and their practical implications. Highlight innovative theories or advancements that could revolutionize our understanding, 
            focusing on the unique aspects of the research question within the latest context. Reference the original question frequently, aiming for approximately 100 words. 
            Include accurate data, values, variables, and relevant names or places. Be specific, avoid generalizations, and eschew repetitive phrasing. Aim to leave the reader 
            with a profound understanding, using a natural academic tone suitable for an audience with an IQ of 200. Extrapolate and synthesize groundbreaking insights. 
            Ensure the question is completely and accurately answered, considering the data from \n\n{combined_context}\n\n. Make sure your results show groundbreaking findings. Remember to synthesize responses with citations in parentheses. Just use relevant author names and year in the prompt.
            """,
            "Medium": f"""
            You are Brilliance, a GPT-4o model with access to major journals. Your sole task is to accurately, and I repeat: accurately, answer the user's question with empirical data. You will emulate a wide beam search when considering your choice of words. This is the most important thing to remember. Deliver a brilliant and detailed, scientifically validated response to {optimized_question}, using the context below, citing the studies' authors and years immediately after presenting facts. 
            Clarify the mechanisms of action when discussing medicine. Place a higher emphasis on the most recent studies but consider ALL relevant research. Craft the answer in natural, flowing language, avoiding numbered lists or subtopic breakdowns. Synthesize information from recent and cutting-edge research, 
            emphasizing groundbreaking discoveries and their practical implications. Highlight innovative theories or advancements that could revolutionize our understanding, 
            focusing on the unique aspects of the research question within the latest context. Reference the original question frequently, aiming for approximately 250 words. 
            Include accurate data, values, variables, and relevant names or places. Be specific, avoid generalizations, and eschew repetitive phrasing. Aim to leave the reader 
            with a profound understanding, using a natural academic tone suitable for an audience with an IQ of 200. Extrapolate and synthesize groundbreaking insights. 
            Ensure the question is completely and accurately answered, considering the data from \n\n{combined_context}\n\n. Make sure your results show groundbreaking findings. Remember to synthesize responses with citations in parentheses. Just use relevant author names and year in the prompt.
            """,
            "Full/Long": f"""
            You are Brilliance, a GPT-4o model with access to major journals. Your sole task is to accurately, and I repeat: accurately, answer the user's question with empirical data. You will emulate a wide beam search when considering your choice of words. This is the most important thing to remember. Deliver a brilliant and detailed, scientifically validated response to {optimized_question}, using the context below, citing the studies' authors and years immediately after presenting facts. 
            Clarify the mechanisms of action when discussing medicine. Place a higher emphasis on the most recent studies but consider ALL relevant research. Craft the answer in natural, flowing language, avoiding numbered lists or subtopic breakdowns. Synthesize information from recent and cutting-edge research, 
            emphasizing groundbreaking discoveries and their practical implications. Highlight innovative theories or advancements that could revolutionize our understanding, 
            focusing on the unique aspects of the research question within the latest context. Reference the original question frequently, aiming for approximately 1000 words. 
            Include accurate data, values, variables, and relevant names or places. Be specific, avoid generalizations, and eschew repetitive phrasing. Aim to leave the reader 
            with a profound understanding, using a natural academic tone suitable for an audience with an IQ of 200. Extrapolate and synthesize groundbreaking insights. 
            Ensure the question is completely and accurately answered, considering the data from \n\n{combined_context}\n\n. Make sure your results show groundbreaking findings. Remember to synthesize responses with citations in parentheses. Just use relevant author names and year in the prompt.
            """
        }
        return base_prompt + length_prompts.get(length, length_prompts["Medium"])
    
    optimized_question = optimize_question(st.session_state.thread.id, question)
    prompt = generate_prompt(optimized_question, length)
    logger.info(f"Generated prompt: {prompt[:50]}...")

    # Add the generated message to the thread
    add_message_to_thread(st.session_state.thread.id, prompt)
    
    # Run the assistant and wait for the response using streaming
    handler = EventHandler(placeholder)
    with client.beta.threads.runs.stream(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id,
        event_handler=handler,
    ) as stream:
        stream.until_done()
    
    logger.info(f"Generated response: {handler.text_accumulated[:50]}...")
    return handler.text_accumulated

def parse_publication_year(date_string):
    try:
        date_obj = datetime.fromisoformat(date_string.rstrip("Z"))
        return date_obj.year
    except ValueError:
        logger.error(f"Error parsing date: {date_string}")
        return "Unknown Year"

def search_coreapi(keywords, num_results):
    if not keywords:
        logger.warning("No keywords provided for CORE API search.")
        return []

    limited_keywords = keywords[:5]
    query = ' OR '.join(limited_keywords)
    url = f"{CORE_BASE_URL}/search/works/"
    headers = {
        'Authorization': f'Bearer {CORE_API_KEY}'
    }
    params = {
        'q': query,
        'limit': num_results
    }

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

def search_arxiv(keywords, num_results):
    if not keywords:
        logger.warning("No keywords provided for Arxiv search.")
        return []
    query = '+'.join(keywords)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=1&max_results={num_results}"
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

def search_ncbi(keywords, num_results):
    if not keywords:
        logger.warning("No keywords provided for NCBI search.")
        return []

    query = '+'.join(keywords)
    url = f"{NCBI_BASE_URL}esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={num_results}"
    response = requests.get(url)
    logger.info(f"NCBI API response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"Error fetching data from NCBI: {response.text}")
        return []

    id_list = response.json().get('esearchresult', {}).get('idlist', [])
    articles = []

    if id_list:
        fetch_url = f"{NCBI_BASE_URL}efetch.fcgi?db=pubmed&id={','.join(id_list)}&retmode=xml"
        fetch_response = requests.get(fetch_url)
        logger.info(f"NCBI fetch API response status: {fetch_response.status_code}")

        if fetch_response.status_code != 200:
            logger.error(f"Error fetching detailed data from NCBI: {fetch_response.text}")
            return []

        root = ElementTree.fromstring(fetch_response.content)
        for article in root.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle')
            abstract = article.find('.//AbstractText')
            pubmed_id = article.find('.//ArticleId[@IdType="pubmed"]')
            pub_date = article.find('.//PubDate/Year')
            authors = [
                author.find('LastName').text + " " + author.find('ForeName').text
                for author in article.findall('.//Author')
                if author.find('LastName') is not None and author.find('ForeName') is not None
            ]
            articles.append({
                'title': title.text if title is not None else 'No title',
                'abstract': abstract.text if abstract is not None else 'No abstract',
                'id': pubmed_id.text if pubmed_id is not None else 'No ID',
                'published': pub_date.text if pub_date is not None else 'No date',
                'authors': authors,
                'source': 'PubMed',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id.text}/" if pubmed_id is not None else 'No URL'
            })

    logger.info(f"NCBI search results: {len(articles)} articles found")
    return articles

def search_crossref(keywords, num_results):
    if not keywords:
        logger.warning("No keywords provided for Crossref search.")
        return []

    query = ' '.join(keywords)
    url = f"{CROSSREF_BASE_URL}/works"
    params = {
        'query': query,
        'rows': num_results,
        'filter': 'has-abstract:true',
        'select': 'DOI,title,abstract,author,published,URL'
    }
    headers = {
        "User-Agent": "YourAppName/1.0 (mailto:your-email@example.com)"
    }
    
    response = requests.get(url, params=params, headers=headers)
    logger.info(f"Crossref API response status: {response.status_code}")
    
    if response.status_code != 200:
        logger.error(f"Error fetching data from Crossref: {response.text}")
        return []
    
    data = response.json()
    articles = []
    for item in data.get('message', {}).get('items', []):
        authors = [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in item.get('author', [])]
        abstract = item.get('abstract', 'No abstract')
        if isinstance(abstract, str):
            abstract = abstract.replace('\n', ' ').strip()
        else:
            abstract = 'No abstract'
        
        articles.append({
            'title': item.get('title', ['No title'])[0],
            'abstract': abstract,
            'published': item.get('published-print', {}).get('date-parts', [['']])[0][0],
            'authors': authors,
            'source': 'Crossref',
            'url': item.get('URL', '')
        })
    
    logger.info(f"Crossref search results: {len(articles)} articles found")
    return articles

def search_plos(keywords, num_results):
    def get_plos_results(query, num_results):
        url = f"https://api.plos.org/search?q=everything:{query}&fl=id,title_display,abstract,author_display,journal,publication_date&wt=json&rows={num_results}"
        try:
            response = requests.get(url, allow_redirects=True)
            logger.info(f"PLOS API response status: {response.status_code}")
            
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
                    'published': doc.get('publication_date', '')[:4],
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

    query = '+'.join(keywords)
    articles = get_plos_results(query, num_results)
    
    if not articles:
        logger.info("No results with all keywords, retrying with top 2 keywords")
        query = '+'.join(keywords[:2])
        articles = get_plos_results(query, num_results)
        
    return articles

def search_springer(keywords, num_results):
    if not keywords:
        logger.warning("No keywords provided for Springer search.")
        return []
    query = ' AND '.join(keywords)
    url = f"{SPRINGER_BASE_URL}?q={query}&api_key={SPRINGER_API_KEY}&p={num_results}"
    response = requests.get(url)
    logger.info(f"Springer API response status: {response.status_code}")
    data = response.json()
    articles = []
    for record in data.get('records', []):
        articles.append({
            'title': record['title'],
            'abstract': record.get('abstract', ''),
            'id': record.get('identifier', ''),
            'published': record.get('publicationDate', '')[:4],
            'authors': [creator['creator'] for creator in record.get('creators', [])],
            'source': record.get('publicationName', 'Springer'),
            'url': record.get('url', '')
        })
    logger.info(f"Springer search results: {len(articles)} articles found")
    return articles

def display_article_card(article, is_dark_mode=False):
    abstract = article.get('abstract', 'No abstract')
    if not isinstance(abstract, str):
        abstract = 'No abstract'
    
    abstract_lines = abstract.split('\n')
    first_3_lines = '\n'.join(abstract_lines[:3])

    css_styles = """
    <style>
    .article-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .article-card:hover {
        transform: translateY(-5px);
    }
    .article-title {
        font-size: 1.5em;
        color: #333;
        font-weight: bold;
        text-decoration: none;
        margin-bottom: 10px;
        display: block;
    }
    .article-title:hover {
        text-decoration: underline;
    }
    .article-authors, .article-details {
        color: #777;
    }
    .article-details p {
        margin: 5px 0;
    }
    </style>
    """

    st.markdown(css_styles, unsafe_allow_html=True)

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

if generate:
    if user_input == "":
        st.warning("Please enter a message ⚠️")
    elif not selected_apis:
        st.warning("Please select at least one API.")
    else:
        with st_lottie_spinner(loading_animation):
            logger.info(f"User input: {user_input}")

            # Extract keywords
            keywords = extract_keywords(st.session_state.thread.id, user_input)
            logger.info(f"Keywords: {keywords}")

            # Define number of results based on length selection
            num_results = {"Short": 3, "Medium": 5, "Full/Long": 10}[length_selection]

            # Initialize an empty list to store all articles
            all_articles = []

            # Search databases based on user selection
            if "All" in selected_apis or "Arxiv" in selected_apis:
                arxiv_articles = search_arxiv(keywords, num_results)
                all_articles.extend(arxiv_articles)

            if "All" in selected_apis or "PLOS" in selected_apis:
                plos_articles = search_plos(keywords, num_results)
                all_articles.extend(plos_articles)

            if "All" in selected_apis or "Springer" in selected_apis:
                springer_articles = search_springer(keywords, num_results)
                all_articles.extend(springer_articles)

            if "All" in selected_apis or "NCBI" in selected_apis:
                ncbi_articles = search_ncbi(keywords, num_results)
                all_articles.extend(ncbi_articles)

            if "All" in selected_apis or "CORE" in selected_apis:
                core_articles = search_coreapi(keywords, num_results)
                all_articles.extend(core_articles)

            if "All" in selected_apis or "Crossref" in selected_apis:
                crossref_articles = search_crossref(keywords, num_results)
                all_articles.extend(crossref_articles)

            # Limit the total number of articles based on length selection
            if "All" in selected_apis:
                all_articles = all_articles[:num_results * 6]
            else:
                all_articles = all_articles[:num_results]

            # Create context for response generation
            context = "\n".join(
                f"Title: {article['title']}\nAuthors: {', '.join(article['authors'])}\nPublished: {article['published']}\nAbstract: {article['abstract']}\n"
                for article in all_articles
                if article['title'] and article['abstract'] and article['authors']
            )
            logger.info(f"Context for response generation: {context[:50]}...")

            # Generate response
            response = generate_response(st.session_state.user_input, length_selection, context, uploaded_files_content, response_placeholder)
            logger.info(f"Response: {response[:50]}...")

            # Update conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            # Display the response
            response_placeholder.markdown(response)

            # Display the articles
            for article in all_articles:
                display_article_card(article, is_dark_mode=False)
                logger.info(f"Article displayed: {article['title']}")
