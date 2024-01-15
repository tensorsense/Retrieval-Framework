import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import OpenAI, AzureOpenAI
from trulens_eval.feedback.provider import AzureOpenAI as TruLensAzureOpenAI, OpenAI as TruLensOpenAI

load_dotenv('.env')

# openai requires OPENAI_API_KEY
# azure requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_DEPLOYMENT_EMBEDDING
OPENAI_PROVIDER = 'azure'

TRULENS_QUESTIONS = [
    "What is the link between AU12 and cognitive load?",
    "How does cognitive load affect facial expressions?",
    "What action units get activated under heavy cognitive load?",
]

# azure openai deployments have strict rate limits
parse_pdfs_num_workers = 1
summarize_pdfs_num_workers = 1


# TODO: use 1 library?
if OPENAI_PROVIDER == 'openai':
    openai_model = OpenAI()
    openai_vision_model = openai_model # TODO: check whether same
    openai_chat_model = ChatOpenAI(temperature=0)
    openai_emb_model = OpenAIEmbeddings()
    openai_trulens_model = TruLensOpenAI()
if OPENAI_PROVIDER == 'azure':
    chat_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_CHAT')
    emb_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING')
    vision_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_VISION')

    openai_model = AzureOpenAI(azure_deployment=chat_deployment)
    openai_vision_model = AzureOpenAI(azure_deployment=vision_deployment)
    openai_chat_model = AzureChatOpenAI(deployment_name=chat_deployment, temperature=0)
    openai_emb_model = AzureOpenAIEmbeddings(deployment=emb_deployment)
    openai_trulens_model = TruLensAzureOpenAI(deployment_name=chat_deployment)

MATHPIX_APP_ID = os.getenv('MATHPIX_APP_ID')
MATHPIX_APP_KEY = os.getenv('MATHPIX_APP_KEY')

PDF_PARSE_TIMEOUT = 60 # seconds