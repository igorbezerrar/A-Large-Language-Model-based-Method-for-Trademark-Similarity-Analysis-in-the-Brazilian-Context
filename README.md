# A Large Language Model-based Method for Trademark Similarity Analysis in the Brazilian Context.

This repository contains the code and models developed for the article research titled "A Large Language Model-based Method for Trademark Similarity Analysis in the Brazilian Context." This study proposes an innovative method utilizing Large Language Models (LLMs) to classify and explain the similarity between word marks, adhering to the criteria established by the Brazilian National Institute of Industrial Property (INPI).

## Project Overview

The increasing volume of trademark applications at INPI presents significant challenges, including prolonged processing times and inconsistencies in decisions. To address these issues, this project introduces an automated method structured into two main components:

1.  **Classification Model:** Identifies conflicts between trademarks with high accuracy.
2.  **Explanation Model:** Provides detailed justifications for similarity or dissimilarity, based on INPI criteria (phonetic, ideological, distinctive, and market-related aspects).

The models were developed and evaluated using a real-world dataset extracted from INPI official publications. The results demonstrate high performance (accuracy ≈99%, F1-score >98%, AUC >99%) and expert-rated clarity of explanations.

## Repository Structure

*   `classifier_model.ipynb`: Jupyter notebook containing the code for the trademark classification model.
*   `explainer_model.ipynb`: Jupyter notebook containing the code for the trademark explanation model.
*   `base_de_dados/dataset_llm.jsonl`: (Expected) The dataset used for training and evaluation.

## Setup and Installation

To run the notebooks in this repository, you need to set up a Python environment and install the necessary libraries. It is highly recommended to use a virtual environment.

### 1. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

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

## Usage

Once the dependencies are installed and the dataset is in place, you can open and run the Jupyter notebooks:

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Open the Notebooks:**
    Navigate to `classifier_model.ipynb` and `explainer_model.ipynb` in your Jupyter interface.

3.  **Run Cells:**
    Execute the cells sequentially in each notebook to replicate the training and evaluation processes for the classification and explanation models, respectively.

## Models

The notebooks utilize various open-source LLMs for fine-tuning, primarily through the `unsloth` library. The models evaluated include:

*   `unsloth/llama-3-8b-bnb-4bit`
*   `unsloth/Qwen2-7B-bnb-4bit`
*   `unsloth/gemma-2-9b-it-bnb-4bit`
*   `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`
*   `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
*   `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`

Users can modify the `models_list` variable in the notebooks to experiment with different LLMs.

## Contributing

Contributions to this project are welcome. Please feel free to open issues or submit pull requests.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/legalcode).

Attribution 4.0 International
CC BY 4.0
Legal Code

