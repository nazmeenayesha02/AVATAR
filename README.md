# Nazmeen Ayesha - Real-World Problem to Verified Action Assistant

Nazmeen Ayesha is a Gemini-powered **Action Assistant** built with **Python, FastAPI, LangGraph, LangChain, Google Gemini, Tavily, ChromaDB, and SQLite** for societal benefit.

It supports real-time streaming chat, document uploads, retrieval-augmented generation (RAG), web search, conversation memory, and a simple web UI.

---

## Features

* Chat with an AI agent powered by Google Gemini
* Stream responses in real time
* Upload documents such as PDF, DOCX, TXT, MD, PY, and CSV
* Use uploaded files as context through RAG
* Search the web with Tavily for current information
* Store and recall conversation history
* Simple FastAPI-based web interface
* Docker-ready deployment
* AWS CI/CD support using GitHub Actions, ECR, and EC2

---

## Project Overview

This project combines:

* **FastAPI** for the backend server and API endpoints
* **Jinja2** for rendering the frontend UI
* **LangGraph** for agent orchestration
* **LangChain** for tools, messages, and RAG workflow
* **Google Gemini** as the LLM provider
* **Tavily** for web search
* **ChromaDB** for vector search over uploaded documents
* **SQLite** for conversation and persistence
* **Docker** for containerized deployment

---

## Prerequisites

Make sure you have the following installed:

* Python 3.11
* pip or conda
* Git
* Google API key for Gemini
* Tavily API key for web search

Optional for deployment:

* Docker
* AWS account
* Amazon ECR repository
* EC2 instance
* GitHub Actions self-hosted runner

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PranavGPT/PranavGPT.git
```

### 2. Navigate to the project directory

```bash
cd PranavGPT
```

### 3. Create a virtual environment

Using conda:

```bash
conda create -n pranavgpt python=3.11 -y
```

### 4. Activate the virtual environment

```bash
conda activate pranavgpt
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root directory.

```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash

TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=pranavgpt
```

If you do not want to use LangSmith tracing, keep:

```env
LANGSMITH_TRACING=false
```

---

## Run Locally

Start the FastAPI app:

```bash
python app.py
```

The app will be available at:

```text
http://127.0.0.1:8000
```

---

## Project Structure

```text
PranavGPT/
│
├── app.py                  # FastAPI app and streaming chat endpoints
├── agent.py                # LangGraph agent setup and tool orchestration
├── database.py             # Conversation and persistence logic
├── rag.py                  # Document ingestion and RAG logic
├── tools.py                # Agent tools such as web search, memory, and RAG
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── .dockerignore           # Docker ignore rules
│
├── templates/
│   └── index.html          # Frontend UI
│
├── uploads/                # Uploaded documents
├── data/                   # SQLite database and app data
└── chroma_db/              # ChromaDB vector database storage
```

---

## Docker Deployment

### 1. Build the Docker image

```bash
docker build -t pranavgpt .
```

### 2. Run the Docker container

```bash
docker run -d \
  --name pranavgpt \
  --restart always \
  -p 8000:8000 \
  --env-file .env \
  pranavgpt
```

The app will be available at:

```text
http://localhost:8000
```

---

## AWS CI/CD Deployment with GitHub Actions

This project can be deployed to AWS using:

* GitHub Actions
* Amazon ECR
* Amazon EC2
* Docker
* GitHub self-hosted runner

---

### 1. Create an IAM User

Create an IAM user for deployment and attach the following policies:

```text
AmazonEC2ContainerRegistryFullAccess
AmazonEC2FullAccess
```

You can also use a more restricted custom IAM policy for production.

---

### 2. Create an ECR Repository

Create an Amazon ECR repository.

For GitHub Secrets, save the repository name:

```text
ECR_REPO=pranavgpt
```

---

### 3. Create an EC2 Instance

Create an Ubuntu EC2 instance.

Recommended inbound security group rule:

```text
Type: Custom TCP
Port: 8000
Source: 0.0.0.0/0
```

---

### 4. Install Docker on EC2

Connect to your EC2 instance and run:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

Add the Ubuntu user to the Docker group:

```bash
sudo usermod -aG docker ubuntu
newgrp docker
```

Check Docker:

```bash
docker --version
```

---

### 5. Configure EC2 as a GitHub Self-Hosted Runner

Go to your GitHub repository:

```text
Settings → Actions → Runners → New self-hosted runner
```

Select Linux and follow the commands shown by GitHub.

After setup, start the runner:

```bash
./run.sh
```

For production, you can configure the runner as a service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## GitHub Secrets

Add the following secrets in your GitHub repository:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO

GOOGLE_API_KEY
GOOGLE_MODEL
TAVILY_API_KEY
LANGSMITH_TRACING
LANGSMITH_ENDPOINT
LANGSMITH_API_KEY
LANGSMITH_PROJECT
```

Path:

```text
GitHub Repository → Settings → Secrets and variables → Actions → New repository secret
```

Example:

```text
AWS_DEFAULT_REGION=us-east-1
ECR_REPO=pranavgpt
GOOGLE_MODEL=gemini-2.5-flash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=pranavgpt
```

---

## GitHub Actions Workflow

Create this file:

```text
.github/workflows/cicd.yaml
```

This workflow will:

1. Build your Docker image
2. Push the image to Amazon ECR
3. Pull the latest image on EC2
4. Stop the old container
5. Run the new container

---

## Usage

After running locally or deploying to AWS:

1. Open the app in your browser.
2. Start chatting with the AI assistant.
3. Upload documents to use them as context.
4. Ask questions about uploaded files.
5. Ask current-information questions to trigger web search.
6. Continue conversations with saved chat history.

---

## Example Questions

```text
Summarize the uploaded PDF.
```

```text
Search the web for the latest AI agent news.
```

```text
Based on my uploaded document, what are the key points?
```

```text
Calculate 125 * 48 / 6.
```

---

## Notes

* Do not commit your `.env` file to GitHub.
* Keep API keys inside GitHub Secrets for deployment.
* For production, avoid using `reload=True` in Uvicorn.
* Make sure port `8000` is open in your EC2 security group.
* Rotate any API keys that were accidentally exposed publicly.

---

## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

---

## License

This project is open source. Please check the repository license for usage terms.
