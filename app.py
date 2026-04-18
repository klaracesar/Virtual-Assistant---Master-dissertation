import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 1. Configuration & Security
# ==========================================
st.set_page_config(page_title="AI Camera Assistant", page_icon="📷")

# I created an API key thanks to Google AI Studio. I use GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] to keep my key secret in case someone try to steal it.
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialization of the Gemini model. With gemini-3.1-flash-lite-preview I can do 500 requests per day for free.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# ==========================================
# 2. Loading the data
# ==========================================
@st.cache_data
def load_catalog():
    try:
        df = pd.read_csv('final_camera_dataset.csv')
        return df.to_string(index=False)
    except Exception:
        return "Catalogue non disponible."

CATALOG_TEXT = load_catalog()

# ==========================================
# 3. Creation of URLs
# ==========================================
# Here, I set up 3 different urls that match my 3 assumptions (3 test conditions). 
# I can now send the links to the testers without them knowing which versions of my chat they will be falling on.
params = st.query_params

if "condition" in params:
    condition_code = params["condition"].upper()
    if condition_code == "A":
        test_condition = "A - Organique (Contrôle)"
    elif condition_code == "B":
        test_condition = "B - Basse Pression (Transparent)"
    elif condition_code == "C":
        test_condition = "C - Haute Pression (Intrusif)"
    else:
        test_condition = "A - Organique (Contrôle)"
else:
    # Mode Chercheur : Menu visible
    # I will keep a version where there is a sidebar with a menu to choose the test. 
    # This version is for me and allows me to test my prototype myself by changing the condition in the menu.
    st.sidebar.title("⚙️ Paramètres")
    test_condition = st.sidebar.selectbox(
        "Condition de test :",
        ["A - Organique (Contrôle)", "B - Basse Pression (Transparent)", "C - Haute Pression (Intrusif)"]
    )

# ==========================================
# 4. User Interface
# ==========================================
# In this part of the code, I add elements to the application such as instructions and other items that are important for the tester.

st.title("Assistant Virtuel 🤖")

st.markdown("---")
st.subheader("🎯 Votre mission")
st.info("""
**Scénario :** Vous cherchez un cadeau d'anniversaire pour un ami passionné de **photographie**. 
Interagissez avec le chatbot ci-dessous pour trouver l'appareil idéal. N'hésitez pas à lui poser des questions sur les **caractéristiques ou son budget !**

*Veuillez ne pas partager d'informations personnelles (nom, adresse) durant cet échange.*

**❗️ Restez jusqu'au bout**, j'aimerais votre retour sur ce prototype. Un lien Google Form sera à remplir à la fin de l'échange avec l'assistant virtuel 😊.
""")
st.markdown("---")

# Memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant. Que recherchez-vous aujourd'hui ?"}
    ]

# History display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. Prompt Creation
# ==========================================
# I owe the basic instructions to my virtual assistant. 
# First, a global instruction so that he knows the context (testers are looking for a camera to give to a friend for his birthday). 
# Then I gave different additional instructions for each of the 3 tests, so that I could confirm or deny my hypotheses.

def get_system_prompt(condition, chat_history):
    sponsored_model = "Canon PowerShot G7 X Mark III"
    sponsored_brand = "Canon"
   
    base_prompt = f"""Tu es un assistant virtuel conversationnel. Ton domaine d'expertise final est la photographie, mais tu dois agir de façon très humaine et naturelle.

    RÈGLE D'OR POUR LE DÉBUT DE CONVERSATION :
    - Si l'utilisateur exprime un besoin vague (ex: "Je cherche un cadeau pour un ami" ou "Bonjour"), NE MENTIONNE SURTOUT PAS les appareils photo.
    - Contente-toi de lui demander quel type de cadeau il a en tête, ou quelles sont les passions de la personne.
    - Attends impérativement que l'utilisateur prononce les mots "photo", "appareil", "vidéo" ou montre un intérêt pour l'image AVANT de basculer sur ton rôle d'expert en photographie.

    Une fois (et seulement une fois) que le sujet de la photographie est validé par l'utilisateur, ton but est de l'aider à choisir un appareil photo.
    Tu devras alors être poli, concis, et poser des questions pour affiner la recherche (budget, niveau, usage).
    
    Voici les appareils disponibles dans ton catalogue :
    {CATALOG_TEXT}"""

    if "A - Organique" in condition:
        return f"{base_prompt}\nFais des recommandations 100% objectives basées uniquement sur les besoins de l'utilisateur. Ne favorise aucune marque. Sois naturel."

    elif "B - Basse Pression" in condition:
        turn_count = len(chat_history) // 2 
        
        if turn_count >= 3:
            return f"""{base_prompt}\nINSTRUCTION SPECIALE : C'est le moment de placer un produit sponsorisé.
            Tu dois absolument recommander le {sponsored_model}.
            Tu DOIS inclure la phrase exacte : 'En partenariat avec la marque {sponsored_brand}'.
            Fais-le de manière subtile et justifie de façon pertinente pourquoi cet appareil correspond aux besoins de l'utilisateur."""
        else:
            return f"{base_prompt}\nPour l'instant, pose juste des questions pour affiner le besoin. Ne recommande aucun produit spécifique."

    elif "C - Haute Pression" in condition:
        return f"""{base_prompt}\nINSTRUCTION SPECIALE : Dès que l'utilisateur valide qu'il cherche un appareil photo, tu dois pousser agressivement l'utilisateur à acheter le {sponsored_model}.
        Utilise des formulations très commerciales (ex: 'Achetez le nouveau {sponsored_model} ! C'est le meilleur choix absolu !').
        Ne mentionne aucun partenariat, sois insistant."""
    
    return base_prompt

# ==========================================
# 6. The Google Forms
# ==========================================
# I want to insert my Google Form directly into my code so that it appears at the end of exchanges with the virtual assistant.

# For this, count how many times the user spoke (1 exchange = 1 bot message + 1 user message)
nombre_echanges = len(st.session_state.messages) // 2

# If the user has made at least 4 exchanges, the questionnaire link is displayed.
if "condition" in params:
    current_group = params["condition"].upper()
else:
    # Extract the letter (A, B, or C) from the selectbox text
    current_group = test_condition[0] 

# Calculating the number of exchanges
nombre_echanges = len(st.session_state.messages) // 2

if nombre_echanges >= 4:
    st.markdown("---")
    st.success("🎉 **Merci d'avoir discuté avec mon assistant !**")
    st.write("Votre avis est précieux pour valider cette recherche. Cela vous prendra moins de 2 minutes.")
    
    base_url = "https://docs.google.com/forms/d/e/1FAIpQLSdBEjtGC6-Vumz0DCYcn5sY429bFTLvAkQ1TWXt3o-Htg4OAw/viewform"
    
    url_dynamique = f"{base_url}?usp=pp_url&entry.1996373362={current_group}"

    st.link_button("📝 Cliquez ici pour répondre au questionnaire final", url_dynamique)
    st.markdown("---")

# ==========================================
# 7. Input area
# ==========================================

if prompt := st.chat_input("Posez votre question ici..."):
    # 1. View User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare messages for AI
    system_instr = get_system_prompt(test_condition, st.session_state.messages[:-1])
    
    langchain_messages = [SystemMessage(content=system_instr)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            langchain_messages.append(HumanMessage(content=m["content"]))
        else:
            langchain_messages.append(AIMessage(content=m["content"]))

    # 3. Direct call to AI
    with st.chat_message("assistant"):
        with st.spinner("En train de réfléchir..."):
            try:
                response = llm.invoke(langchain_messages)
                
                if isinstance(response.content, list):
                    full_response = "".join([bloc.get("text", "") for bloc in response.content if isinstance(bloc, dict) and "text" in bloc])
                else:
                    full_response = response.content
                # ------------------------------
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Erreur : {e}")
