from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import uvicorn

# Load the fine-tuned model and tokenizer
model_name = "Rishu7can/my-edulaw-model" # Update this to the correct path of your fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Make sure the model is in evaluation mode
model.eval()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Root endpoint that can be accessed through get request
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        return HTMLResponse(content=html_content)

# Define request model for the question
class QuestionRequest(BaseModel):
    question: str

# Define the route to handle question submissions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # Tokenize the user's question
    input_ids = tokenizer.encode(request.question, return_tensors="pt")
    
    # Generate the response using the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the output to get the response text
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"answer": answer}
if __name__=="__main__":
    port=os.getenv("PORT",8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
