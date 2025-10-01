# 🤖 AI Customer Support Bot with Document Training & Feedback Loop

An intelligent customer support chatbot that learns from FAQ documents, answers queries using NLP models, and iteratively improves responses through simulated feedback mechanisms.

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/NG-2004/Serri)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Live Demo

Try the bot now: **[https://huggingface.co/spaces/NG-2004/Serri](https://huggingface.co/spaces/NG-2004/Serri)**

## 📋 Overview

This project implements an agentic workflow that:
- Trains on provided documents (FAQ, manuals, policies)
- Uses semantic search to find relevant information
- Generates accurate answers using transformer-based Q&A models
- Improves responses through a feedback loop (max 2 iterations)
- Logs all decisions and actions for transparency
- Handles out-of-scope queries gracefully

## ✨ Features

- **📄 Document Training**: Ingests and processes FAQ documents
- **🧠 Semantic Search**: Uses sentence embeddings for intelligent context retrieval
- **💬 Question Answering**: Employs DistilBERT for extractive Q&A
- **🔄 Feedback Loop**: Simulates user feedback and adjusts responses accordingly
- **📝 Comprehensive Logging**: Tracks all queries, decisions, and iterations
- **🌐 Gradio Interface**: Interactive web UI with multiple tabs
- **🚀 HuggingFace Deployment**: Deployed on free CPU tier

## 🛠️ Technology Stack

- **Python 3.8+**
- **Transformers** (Hugging Face) - Question answering
- **Sentence Transformers** - Semantic search and embeddings
- **PyTorch** - Deep learning backend
- **Gradio** - Web interface
- **Logging** - Decision tracking

## 🔧 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-support-bot.git
cd customer-support-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
transformers==4.35.0
sentence-transformers==2.2.2
torch==2.1.0
gradio==4.8.0
```

## 🚀 Usage

### Run Jupyter Notebook

```bash
jupyter notebook Support_bot.ipynb
```

### Run as Python Script

Create `app.py` from the notebook cells and run:

```bash
python app.py
```

The Gradio interface will launch at `http://127.0.0.1:7860`

## 📂 Project Structure

```
customer-support-bot/
├── Support_bot.ipynb          # Main development notebook
├── faq.txt                    # Sample FAQ document
├── support_bot_log.txt        # Generated log file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── app.py                     # Standalone Python script (optional)
```

## 🎯 How It Works

### 1. Document Processing
- Loads FAQ document and splits into sections
- Creates vector embeddings for each section using `all-MiniLM-L6-v2`

### 2. Query Handling
- User submits a question
- Semantic search finds the most relevant section (cosine similarity)
- Relevance threshold: 0.3 (configurable)

### 3. Answer Generation
- Uses `distilbert-base-uncased-distilled-squad` for extractive Q&A
- Extracts answer from relevant context
- Returns confidence score

### 4. Feedback Loop
- Simulates user feedback: "good", "too vague", "not helpful"
- **Too vague** → Adds more context from document
- **Not helpful** → Rephrases query and retries
- **Good** → Accepts response
- Maximum 2 iterations per query

### 5. Logging
All actions logged to `support_bot_log.txt`:
- Query received
- Relevant section found (with similarity score)
- Generated answer (with confidence)
- Feedback received
- Adjustments made

## 📊 Example Queries

```python
queries = [
    "How do I reset my password?",
    "What is your refund policy?",
    "How can I contact support?",
    "What are the shipping options?",
    "How do I track my order?",
    "What payment methods do you accept?"
]
```

## 🎨 Gradio Interface

The web interface includes three tabs:

1. **💬 Chat**: Direct question answering
2. **🔄 Feedback Simulation**: Shows iterative improvement process
3. **📊 About**: Information about the bot and technology

## 📈 Model Performance

| Model | Purpose | Size | Speed |
|-------|---------|------|-------|
| DistilBERT | Question Answering | ~250MB | Fast |
| MiniLM-L6-v2 | Sentence Embeddings | ~80MB | Very Fast |

## 🔮 Future Enhancements

### Planned Improvements

1. **🦙 LLaMA 2 Integration**
   - Replace DistilBERT with LLaMA 2 for generative responses
   - Enable multi-turn conversations
   - Better context understanding
   - More natural, detailed answers

2. **🤖 GPT-2 Alternative**
   - Lightweight generative model option
   - Creative response generation
   - Abstractive summarization
   - Paraphrasing capabilities

3. **Additional Features**
   - Multi-document support
   - Real user feedback integration
   - Conversation history tracking
   - Fine-tuning on domain-specific data
   - Multi-language support
   - Confidence threshold tuning
   - RAG (Retrieval-Augmented Generation) pipeline
   - Vector database integration (Pinecone, Weaviate)

## 🐛 Known Issues

- First run downloads models (~500MB) - takes 2-3 minutes
- CPU inference is slower than GPU
- Limited to extractive answers (no generation)
- Fixed relevance threshold (0.3)

## 📝 Development Decisions

### Why DistilBERT?
- Lightweight (40% smaller than BERT)
- Fast inference on CPU
- Good accuracy for Q&A tasks
- Well-documented and supported

### Why Sentence Transformers?
- Semantic search outperforms keyword matching
- Handles paraphrased questions effectively
- Fast embedding generation
- Easy to use and deploy

### Why Gradio?
- Quick prototyping
- No web development required
- Built-in sharing capabilities
- Clean, modern UI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Nanda Gopal**

- HuggingFace: [@NG-2004](https://huggingface.co/NG-2004)
- GitHub: [@NANDAGOPALNG](https://github.com/NANDAGOPALNG)

## 🙏 Acknowledgments

- Assignment provided by Serri AI
- Built with Hugging Face Transformers
- Deployed on Hugging Face Spaces
- Inspired by modern customer support systems

## 📧 Contact

For questions or support, please contact:
- Email: nandagopalng@gmail.com
- HuggingFace Space: [Live Demo](https://huggingface.co/spaces/NG-2004/Serri)

---

**Note**: This is a demonstration project for educational purposes. For production use, consider implementing proper authentication, rate limiting, and error handling.

## 🎓 Assignment Context

This project was developed as part of an ML internship assignment requiring:
- Document-based training
- NLP model integration
- Feedback loop implementation
- Logging and transparency
- Graceful error handling
- Interactive deployment

All requirements have been successfully implemented and deployed.
