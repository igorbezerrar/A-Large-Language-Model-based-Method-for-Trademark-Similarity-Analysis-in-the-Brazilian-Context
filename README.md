# 🧠 A Large Language Model-based Method for Trademark Similarity Analysis in the Brazilian Context.

This repository contains the code and models developed for the research article titled "A Large Language Model-based Method for Trademark Similarity Analysis in the Brazilian Context." This study proposes an innovative method utilizing Large Language Models (LLMs) to classify and explain the similarity between word marks, adhering to the criteria established by the Brazilian National Institute of Industrial Property (INPI).

## 🤖 Access our published paper

Our research article, "A Large Language Model-based Method for Trademark Similarity Analysis in the Brazilian Context," has been submitted to the **World Patent Information** journal. We will update this section with the publication details once it is officially accepted and published.

## 💡 Project Overview

The increasing volume of trademark applications at INPI presents significant challenges, including prolonged processing times and inconsistencies in decisions. To address these issues, this project introduces an automated method structured into two main components:

1.  **Classification Model:** Identifies conflicts between trademarks with high accuracy.
2.  **Explanation Model:** Provides detailed justifications for similarity or dissimilarity, based on INPI criteria (phonetic, ideological, distinctive, and market-related aspects).

The models were developed and evaluated using a real-world dataset extracted from INPI official publications. The results demonstrate high performance (accuracy ≈99%, F1-score >98%, AUC >99%) and expert-rated clarity of explanations.

This project:

✔️ Uses **QLoRA and Unsloth** for memory-efficient fine-tuning of LLMs  
✔️ Implements a **two-model pipeline**: a **classifier** and an **explainer**  
✔️ Trains and evaluates multiple LLMs (**Mistral-Nemo**, **Qwen2-7B**, **Llama-3-8B**)  
✔️ Follows INPI's criteria: **Phonetic**, **Ideological**, **Distinctiveness**, and **Market Proximity**  
✔️ Includes **expert-validated explanation reports** for supervised evaluation  
✔️ Provides **classification metrics (Accuracy, F1, AUC)** and **qualitative expert feedback**

## 🚀 Technologies Used

- **Transformers** 🤗 – NLP model loading and inference  
- **Unsloth** ⚡ – Optimized fine-tuning for LLaMA and similar models  
- **PEFT (QLoRA)** 🛠️ – Memory-efficient parameter tuning  
- **scikit-learn** 📊 – Metrics and classification reports  
- **Matplotlib & Seaborn** 📉 – Visualization (Confusion Matrices, ROC Curves, Boxplots)  
- **Pandas & NumPy** 🏗️ – Data manipulation  
- **Datasets** 📚 – Dataset management  

---


## 📂 Project Structure

### **1️⃣ Dataset Preparation**
- Uses the **Conflicting Marks Archive Dataset (CMAD)**  
- Contains real-world trademark conflict cases from INPI  

### **2️⃣ Classifier Fine-Tuning**
- Fine-tunes LLMs using **binary-labeled pairs** (Similar / Different)  
- Applies **QLoRA + Unsloth** for low-resource training  

### **3️⃣ Explanation Prompt Engineering**
- Builds **few-shot prompts** using **expert-reviewed examples**  
- Follows **INPI's four similarity criteria** for justification generation  

### **4️⃣ Evaluation Pipeline**
- Calculates **Accuracy**, **Precision**, **Recall**, **F1-score**, **AUC**  
- Generates **Confusion Matrices** and **ROC Curves**  
- Collects **expert feedback** on explanation quality (Likert scale 0–5)  

### **5️⃣ Error Analysis & Visualization**
- Provides detailed **error analysis**, identifying common misclassification patterns  
- Generates **boxplots** for explanation score distributions by model  

## 📂 Repository Structure

*   `classifier_model.ipynb`: Jupyter notebook containing the code for the trademark classification model.
*   `explainer_model.ipynb`: Jupyter notebook containing the code for the trademark explanation model.
*   `base_de_dados/dataset_llm.jsonl`: (Expected) The dataset used for training and evaluation.

## ⚙️ Setup and Installation

To run the notebooks in this repository, you need to set up a Python environment and install the necessary libraries. It is highly recommended to use a virtual environment.

### 1. Create a Virtual Environment (Optional but Recommended)

It is highly recommended to use Python 3.10.12 for this project due to compatibility with `torch.compile` and `unsloth`. Follow these steps to create a virtual environment with Python 3.10.12:

1.  **Check Python 3.10.12 availability:**
    ```bash
    python3.10 --version
    ```
    If Python 3.10.12 is not found, you may need to install it first. On Debian/Ubuntu systems, you can install it using:
    ```bash
    sudo apt update
    sudo apt install -y python3.10-venv
    ```
2.  **Create the virtual environment:**
    ```bash
    python3.10 -m venv venv_py310
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv_py310/bin/activate
    ```

Once activated, your terminal prompt should change to indicate that you are in the `venv_py310` environment.

### 2. Install Dependencies

The core models in these notebooks leverage `unsloth` for efficient fine-tuning of Large Language Models. The following libraries are required:

```bash
pip install torch
pip install xformers==0.0.27
pip install scikit-learn
pip install datasets
pip install trl
pip install pandas
pip install numpy
pip install transformers
pip install matplotlib
pip install huggingface_hub
pip install bitsandbytes
pip install tyro
pip install sentencepiece
pip install tqdm
pip install psutil
pip install wheel
pip install protobuf
pip install hf_transfer

# Install unsloth from source for the latest features and compatibility
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 3. Data Preparation

The notebooks expect a dataset file named `dataset_llm.jsonl` located in a directory named `base_de_dados` at the root of the repository. Please ensure this file and directory structure are in place before running the notebooks.

```
./
├── classifier_model.ipynb
├── explainer_model.ipynb
└── base_de_dados/
    └── dataset_llm.jsonl
```

## 🚀 Usage

Once the dependencies are installed and the dataset is in place, you can open and run the Jupyter notebooks:

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Open the Notebooks:**
    Navigate to `classifier_model.ipynb` and `explainer_model.ipynb` in your Jupyter interface.

3.  **Run Cells:**
    Execute the cells sequentially in each notebook to replicate the training and evaluation processes for the classification and explanation models, respectively.

## 🧠 Models

The notebooks utilize various open-source LLMs for fine-tuning, primarily through the `unsloth` library. The models evaluated include:

*   `unsloth/llama-3-8b-bnb-4bit`
*   `unsloth/Qwen2-7B-bnb-4bit`
*   `unsloth/gemma-2-9b-it-bnb-4bit`
*   `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`
*   `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
*   `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`

Users can modify the `models_list` variable in the notebooks to experiment with different LLMs.

## 👏 Contributing

We welcome contributions!  
If you spot any issues or have suggestions for improvement, feel free to open an issue or pull request.

For questions, contact:  
[![Gmail Badge](https://img.shields.io/badge/-igor.reis@ifpi.edu.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:igor.reis@ifpi.edu.br)](mailto:igor.reis@ifpi.edu.br)
[![Gmail Badge](https://img.shields.io/badge/-ariel.teles@ifma.edu.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:ariel.teles@ifma.edu.br)](mailto:ariel.teles@ifma.edu.br)  

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/legalcode).

Attribution 4.0 International
CC BY 4.0
Legal Code

