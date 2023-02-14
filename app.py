from transformers import AutoModelForCausalLM, AutoTokenizer

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, tokenizer
    model_name = "bigscience/bloomz-7b1"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model, tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(inputs)

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the results as a dictionary
    return result
