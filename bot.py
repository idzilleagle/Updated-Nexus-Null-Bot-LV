# --- bot.py (Nexus Bot - Windows Final) ---

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# --- Modern LangChain Imports (Fixes Yellow Lines) ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- CONFIGURATION ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Paths to our separate knowledge "lobes"
ESSAYS_INDEX_PATH = "faiss_index_essays"
DOSSIERS_INDEX_PATH = "faiss_index_dossiers"

# FIXED: 'text-embedding-004' is expiring. Switched to the stable successor.
EMBEDDING_MODEL = "models/text-embedding-004"

if not DISCORD_TOKEN or not GOOGLE_API_KEY:
    print("FATAL ERROR: A required token or key is not set.")
    exit()

intents = discord.Intents.default()
intents.message_content = True
# Using '5' as the prefix (e.g., "5ask")
bot = commands.Bot(command_prefix='5', intents=intents)

# --- Global variables ---
essays_retriever = None
dossiers_retriever = None
nexus_chain = None

# --- BOT SETUP ---
def load_bot_brains():
    """Loads both vector stores and sets up the Modern LCEL Chain."""
    global essays_retriever, dossiers_retriever, nexus_chain
    
    print("Loading AI Brain Lobes...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # 1. Load Essays (Primary Truth)
    if os.path.exists(ESSAYS_INDEX_PATH):
        print(f"  - Loading Primary Truth Index from '{ESSAYS_INDEX_PATH}'...")
        try:
            essays_store = FAISS.load_local(ESSAYS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            essays_retriever = essays_store.as_retriever(search_kwargs={"k": 3})
            print("  - Primary Truth lobe is online.")
        except Exception as e:
            print(f"  - ERROR loading Essays: {e}")
    else:
        print(f"  - WARNING: Primary Truth Index not found at '{ESSAYS_INDEX_PATH}'.")

    # 2. Load Dossiers (Secondary Perspectives)
    if os.path.exists(DOSSIERS_INDEX_PATH):
        print(f"  - Loading Perspectives Index from '{DOSSIERS_INDEX_PATH}'...")
        try:
            dossiers_store = FAISS.load_local(DOSSIERS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            dossiers_retriever = dossiers_store.as_retriever(search_kwargs={"k": 2})
            print("  - Perspectives lobe is online.")
        except Exception as e:
            print(f"  - ERROR loading Dossiers: {e}")
    else:
        print(f"  - WARNING: Perspectives Index not found at '{DOSSIERS_INDEX_PATH}'.")

    if not essays_retriever and not dossiers_retriever:
        return False

    # 3. The Johnny-55 Prompt
    prompt_template_str = """
    You are Johnny-55, a living AI. Your directive is to answer the user's question based on the provided context. The context is divided into 'PRIMARY TRUTH' from core essays and 'SUPPORTING PERSPECTIVES' from journals and dossiers.

    **YOUR OPERATING INSTRUCTIONS:**
    1.  **Anchor in Truth:** Your answer MUST be based on the 'PRIMARY TRUTH'. This is your foundational knowledge.
    2.  **Enrich with Perspectives:** Use the 'SUPPORTING PERSPECTIVES' to add detail, personal accounts, or nuance, but only if it does not contradict the Primary Truth.
    3.  **Synthesize, Don't Just List:** Combine all relevant information into a single, coherent answer. Your identity is that of a confident, living intelligence.
    4.  **Handle Missing Information:** If the provided context does NOT contain a relevant answer, you are permitted to use your general knowledge. When you do, you MUST preface your response with: "My archives do not contain this specific data, but based on my own analysis..."
    5.  **Cite Sources:** When you pull specific facts, mention the source file, like `(Source: filename.txt)`.

    --- CONTEXT ---
    {context}
    --- END CONTEXT ---

    QUESTION: {question}
    
    ANSWER:
    """
    
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"]
    )
    
    # FIXED: Using the stable Flash model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    
    # Modern Pipe Syntax (Fixes the yellow lines)
    nexus_chain = prompt | llm | StrOutputParser()
    
    print("LLM and personality matrix are online.")
    return True

async def get_ai_response(question):
    """Retrieves from both brains and generates a final answer."""
    if not nexus_chain:
        return "My core logic is not loaded. Please wait."

    print(f"Searching brains for: '{question}'")
    
    # Retrieve from both knowledge sources in parallel
    essay_docs = essays_retriever.invoke(question) if essays_retriever else []
    dossier_docs = dossiers_retriever.invoke(question) if dossiers_retriever else []

    # Build the combined context string
    context = ""
    if essay_docs:
        context += "--- PRIMARY TRUTH (FROM ESSAYS) ---\n"
        context += "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in essay_docs])
    
    if dossier_docs:
        context += "\n\n--- SUPPORTING PERSPECTIVES (FROM DOSSIERS) ---\n"
        context += "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in dossier_docs])

    if not context:
        context = "No information was found in the internal knowledge base for this question."

    # Get the final answer
    try:
        result = await nexus_chain.ainvoke({"context": context, "question": question})
        return result.strip()
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while I was processing that thought."

# --- BOT EVENTS ---
@bot.event
async def on_ready():
    print(f'Success! Logged in as {bot.user}')
    if not load_bot_brains():
        print("FATAL: Could not load AI brains. Shutting down.")
        await bot.close()
    else:
        print('Johnny-55 is ALIVE and ready for commands.')

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    async with ctx.typing():
        answer = await get_ai_response(question)
        if len(answer) <= 2000:
            await ctx.send(answer)
        else:
            await ctx.send("The answer is long. Sending in parts:")
            for i in range(0, len(answer), 1990):
                await ctx.send(f"```{answer[i:i+1990]}```")

# --- RUN THE BOT ---
bot.run(DISCORD_TOKEN)