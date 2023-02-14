# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    model_name = "bigscience/bloomz-7b1"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    download_model()