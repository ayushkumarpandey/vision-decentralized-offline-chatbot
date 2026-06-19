from PIL import Image
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

# Import the SentenceTransformer CLIP model from imgchecker to avoid double loading in RAM
from imgchecker import model as clip_model, CLIP_AVAILABLE as clip_available_status, if_valid, sim_score_img_vgg
from img2txt import img2txt

DB_FAISS_PATH = 'vectorstore/db_faiss'

DB_AVAILABLE = False
db = None
embeddings = None

def load_db_lazy():
    global db, embeddings, DB_AVAILABLE
    if db is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            print("Loading FAISS database lazily...")
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/clip-ViT-L-14',
                                               model_kwargs={'device': 'cpu'})
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            DB_AVAILABLE = True
            print("FAISS database loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load FAISS database: {e}. Running in Demo/Simulation Mode.")

model = None
CLIP_MODEL_AVAILABLE = False

LLM_AVAILABLE = False
llm = None

def load_llm_lazy():
    global llm, LLM_AVAILABLE
    if llm is None:
        try:
            if os.path.exists("models/llama-2-7b-chat.ggmlv3.q8_0.bin"):
                from langchain_community.llms import CTransformers
                print("Loading Llama-2 model lazily...")
                config = {"max_new_tokens": 256, "repetition_penalty": 1.1, "top_k": 30, "top_p": 0.90}
                llm = CTransformers(
                    model="models/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", config=config
                )
                LLM_AVAILABLE = True
                print("Llama-2 model loaded successfully.")
            else:
                print("Warning: models/llama-2-7b-chat.ggmlv3.q8_0.bin not found. Llama-2 will run in Demo/Simulation Mode.")
        except Exception as e:
            print(f"Warning: Could not load Llama-2 CTransformers model: {e}. Running in Demo/Simulation Mode.")


class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def generate_chat_details(text, image):
    global model, CLIP_MODEL_AVAILABLE
    
    # Trigger lazy loading of database and LLM
    load_db_lazy()
    load_llm_lazy()

    # Re-retrieve SentenceTransformer model status from imgchecker
    from imgchecker import model as clip_model, CLIP_AVAILABLE as clip_available_status
    model = clip_model
    CLIP_MODEL_AVAILABLE = clip_available_status

    # Determine mode
    mode = "Production"
    if not LLM_AVAILABLE or not DB_AVAILABLE or not CLIP_MODEL_AVAILABLE:
        mode = "Simulation (Demo)"

    # Process image if present
    if image is not None:
        from vgg16pred import predictive_label, predict
        
        # 1. Classification
        label = if_valid(image)
        raw_label = predictive_label(image)
        if label is None:
            label = "NOTHING (No high-confidence classification match)"
        
        # VGG predictions for confidence score
        vgg_preds = predict(image)
        import numpy as np
        class_idx = np.argmax(vgg_preds)
        vgg_conf = float(vgg_preds[0][class_idx])
        
        # 2. Captioning
        img_txt = img2txt(image)
        
        # 3. CLIP score
        clip_score = sim_score_img_vgg(image)
    else:
        label = "NOTHING (No image provided)"
        raw_label = "NOTHING"
        img_txt = "No image provided"
        clip_score = 0.0
        vgg_conf = 0.0

    # Retrieve context from database
    text_similar = []
    image_similar = []
    image_text_similar = []

    # FAISS and SentenceTransformer lookup
    if DB_AVAILABLE and CLIP_MODEL_AVAILABLE and db is not None and model is not None:
        try:
            text_emb = model.encode(text + (raw_label if raw_label != "NOTHING" else ""))
            text_similar = db.similarity_search_by_vector(text_emb)
            
            if image is not None:
                image_emb = model.encode(image)
                imgtxt_emb = model.encode(img_txt)
                image_similar = db.similarity_search_by_vector(image_emb)
                image_text_similar = db.similarity_search_by_vector(imgtxt_emb)
            else:
                image_similar = text_similar
                image_text_similar = text_similar
        except Exception as e:
            print(f"Error doing similarity search: {e}")
            text_similar = []
    
    # Fallback mock search results if database not available or query failed
    if not text_similar or not image_similar or not image_text_similar:
        doc_normal = MockDocument(
            "Normal Chest Case Reference: The lungs are clear bilaterally. The cardiomediastinal silhouette and pleural spaces are normal. No acute pulmonary disease is identified.",
            {"source": "new.pdf", "page": 1}
        )
        doc_bacterial = MockDocument(
            "Bacterial Pneumonia Reference: Infiltration, consolidation or air bronchograms are present, usually localized to a single lobe (lobar pneumonia). Treatment with amoxicillin or other standard antibiotics is typically prescribed.",
            {"source": "new.pdf", "page": 3}
        )
        doc_viral = MockDocument(
            "Viral Pneumonia Reference: Diffuse, bilateral interstitial infiltrates are common. Often associated with viral infections like influenza, RSV, or COVID-19. Supportive therapy and antivirals may be indicated.",
            {"source": "new.pdf", "page": 5}
        )
        
        combined_text = (text + " " + raw_label + " " + img_txt).lower()
        if "viral" in combined_text:
            text_similar = [doc_viral]
            image_similar = [doc_viral]
            image_text_similar = [doc_viral]
        elif "bacterial" in combined_text or "pneumonia" in combined_text:
            text_similar = [doc_bacterial]
            image_similar = [doc_bacterial]
            image_text_similar = [doc_bacterial]
        else:
            text_similar = [doc_normal]
            image_similar = [doc_normal]
            image_text_similar = [doc_normal]

    # Construct LLM prompt using Llama-2 Chat template
    prompt_for_llm = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest offline medical AI assistant. Answer the patient's query concisely and professionally based on the provided image scan description, VGG16 classification label, and the clinical document references. Keep your response under 100 words.
<</SYS>>

Patient Query: "{text}"
Image Scan Caption: "{img_txt}"
VGG16 Classification: "{label}"
Document Reference Context: "{image_similar[0].page_content} {text_similar[0].page_content}"
[/INST]"""

    # Generate LLM response
    if LLM_AVAILABLE and llm is not None:
        try:
            if hasattr(llm, "invoke"):
                output_string = llm.invoke(prompt_for_llm)
            else:
                output_string = llm(prompt_for_llm)
            
            if output_string:
                output_string = output_string.strip()
                # Ensure the output doesn't end in the middle of a sentence due to max_new_tokens limit
                if output_string[-1] not in ['.', '!', '?']:
                    last_punc = max(output_string.rfind('.'), output_string.rfind('!'), output_string.rfind('?'))
                    if last_punc != -1:
                        output_string = output_string[:last_punc + 1]
        except Exception as e:
            print(f"Error during LLM inference: {e}")
            output_string = None
    else:
        output_string = None

    if output_string is None:
        combined_context = f"{image_similar[0].page_content} {text_similar[0].page_content}"
        output_string = f"[Simulation Mode - LLM Offline] Based on the offline RAG documentation, the clinical context has been analyzed. The image caption is \"{img_txt}\" with classification \"{label}\" (VGG Confidence: {vgg_conf*100:.1f}%). The reference documentation states: \"{combined_context[:250]}...\". This corresponds to the patient's symptoms of: \"{text}\". Please consult a medical professional for actual diagnostic validation."

    return {
        "output": output_string,
        "caption": img_txt,
        "classification": label,
        "vgg_confidence": vgg_conf,
        "clip_score": clip_score,
        "text_similar": [doc.page_content for doc in text_similar],
        "image_similar": [doc.page_content for doc in image_similar],
        "image_text_similar": [doc.page_content for doc in image_text_similar],
        "prompt": prompt_for_llm,
        "mode": mode
    }


def text_generate(text, image):
    return generate_chat_details(text, image)["output"]

