# Fine-Tuning GPT-2 for Medical Question Answering using MedQuAD

## 📖 Project Overview
This project aims to fine-tune the GPT-2 model on the MedQuAD (Medical Question-Answer Dataset) to create a domain-specific language model that can accurately answer medical questions. The fine-tuned model will be evaluated using multiple fine-tuning techniques to compare their performance.

### Fine-Tuning Techniques Compared:
1. Vanilla Fine-Tuning: Standard fine-tuning on the MedQuAD dataset.
2. LoRA (Low-Rank Adaptation): Efficient fine-tuning using low-rank adaptation matrices.
3. QLoRA: Combining LoRA with quantized models to reduce memory requirements during fine-tuning.
4. Adapters: Introducing adapter layers to the model for efficient task-specific training.
5. Reinforcement Learning with Human Feedback (RLHF): Fine-tuning the model using human feedback and reward signals.

### Goal:
The goal of this project is to fine-tune GPT-2 to generate accurate medical responses for various medical-related queries. The model's performance will be evaluated on its ability to answer questions from the MedQuAD dataset, and the results from different fine-tuning methods will be compared.




###  Setup - Dockertfile
``` DockerFile
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Expose the necessary port for Jupyter Notebook
EXPOSE 8888

# Set the environment variable for the model
ENV PYTHONUNBUFFERED=1

# Command to start Jupyter Notebook when the container runs
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

```


