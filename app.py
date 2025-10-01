import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import random
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    filename='support_bot_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SupportBotAgent:
    """Intelligent Customer Support Bot"""
    
    def __init__(self, document_path):
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad"
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_text = self.load_document(document_path)
        self.sections = [s.strip() for s in self.document_text.split('\n\n') if s.strip()]
        self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
        logging.info(f"Loaded document: {document_path} with {len(self.sections)} sections")
    
    def load_document(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            return ""
    
    def find_relevant_section(self, query):
        if not self.sections:
            return None
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        logging.info(f"Query: '{query}' | Match score: {best_score:.3f}")
        return self.sections[best_idx] if best_score >= 0.3 else None
    
    def answer_query(self, query):
        context = self.find_relevant_section(query)
        if not context:
            return "I don't have enough information to answer that. Please contact support@example.com or call 1-800-555-1234."
        try:
            result = self.qa_model(question=query, context=context)
            return result["answer"]
        except:
            return "I encountered an error. Please try rephrasing your question."
    
    def get_feedback(self, response):
        feedback = random.choices(["not helpful", "too vague", "good"], weights=[0.2, 0.3, 0.5])[0]
        logging.info(f"Feedback: {feedback}")
        return feedback
    
    def adjust_response(self, query, response, feedback):
        if feedback == "too vague":
            context = self.find_relevant_section(query)
            return f"{response}\n\nAdditional context: {context[:200]}..." if context else response
        elif feedback == "not helpful":
            return self.answer_query(f"{query} Please provide more details.")
        return response
    
    def process_query_with_feedback(self, query):
        if not query.strip():
            return "Please enter a question."
        
        logging.info(f"NEW QUERY: {query}")
        response = self.answer_query(query)
        history = [f"**Initial Response:** {response}"]
        
        for iteration in range(2):
            feedback = self.get_feedback(response)
            if feedback == "good":
                history.append(f"\n**Feedback {iteration + 1}:** ✅ Good")
                break
            history.append(f"\n**Feedback {iteration + 1}:** ⚠️ {feedback.title()}")
            response = self.adjust_response(query, response, feedback)
            history.append(f"**Adjusted Response:** {response}")
        
        return "\n".join(history)
    
    def chat(self, query):
        return self.answer_query(query) if query.strip() else "Please enter a question."

# Initialize bot
bot = SupportBotAgent("faq.txt")

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="AI Support Bot") as demo:
    gr.Markdown(
        """
        # 🤖 AI Customer Support Bot
        ### Ask me anything about our services, policies, and support!
        """
    )
    
    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Your Question", placeholder="How do I reset my password?", lines=2)
                submit_btn = gr.Button("Ask Question", variant="primary")
                gr.Examples(
                    ["How do I reset my password?", "What is your refund policy?", "How can I contact support?"],
                    inputs=query_input
                )
            with gr.Column():
                response_output = gr.Textbox(label="Bot Response", lines=10)
        
        submit_btn.click(bot.chat, inputs=query_input, outputs=response_output)
    
    with gr.Tab("🔄 Feedback Simulation"):
        query_feedback = gr.Textbox(label="Your Question", lines=2)
        feedback_btn = gr.Button("Process with Feedback", variant="primary")
        feedback_output = gr.Textbox(label="Response History", lines=15)
        feedback_btn.click(bot.process_query_with_feedback, inputs=query_feedback, outputs=feedback_output)
    
    with gr.Tab("📊 About"):
        gr.Markdown(
            """
            ## About This Bot
            - 📄 Trained on company FAQ documents
            - 🧠 Uses DistilBERT for Q&A and Sentence Transformers for search
            - 🔄 Improves responses through feedback loops
            - 📝 Logs all decisions for transparency
            
            **Tech Stack**: Python, Transformers, Gradio, HuggingFace
            """
        )

if __name__ == "__main__":
    demo.launch()
