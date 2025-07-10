from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_id = "tiiuae/falcon-rw-1b"  # Free Hugging Face model
client = InferenceClient(model=model_id)

class DietRequest(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    diet_goal: str
    allergies: str = ""

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

@app.post("/diet-plan")
async def generate_diet_plan(request: DietRequest):
    bmi = calculate_bmi(request.height_cm, request.weight_kg)

    prompt = (
        f"Create a 1-day diet plan in table format for a person:\n"
        f"Age: {request.age}, Height: {request.height_cm}cm, "
        f"Weight: {request.weight_kg}kg, BMI: {bmi}, "
        f"Goal: {request.diet_goal}, Allergies: {request.allergies or 'None'}\n"
        f"Format: Meal | Items | Calories (approx)"
    )

    try:
        response = client.text_generation(prompt=prompt, max_new_tokens=300)
        return {"bmi": bmi, "response": response.strip()}
    except Exception as e:
        return {"error": str(e)}
