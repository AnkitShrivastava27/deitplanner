from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import re
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()

# CORS for Flutter/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API setup (must be set in environment)
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

# -----------------------------
# Input model for diet planning
# -----------------------------
class DietRequest(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    diet_goal: str
    allergies: Optional[str] = ""
    budget: Optional[str] = None   # Flexible input (low, medium, $1000, etc.)

# -----------------------------
# Helper: BMI calculation
# -----------------------------
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

# -----------------------------
# Helper: Budget Normalization
# -----------------------------
def normalize_budget(budget: Optional[str]) -> str:
    if not budget:
        return "Not specified"

    b = budget.strip().lower()

    # Handle text categories
    if b in ["low", "cheap", "basic"]:
        return "500 INR (low budget)"
    if b in ["medium", "moderate"]:
        return "1000 INR (medium budget)"
    if b in ["high", "expensive", "premium"]:
        return "2000 INR (high budget)"

    # Handle "under 1000" type input
    if "under" in b:
        numbers = re.findall(r"\d+", b)
        if numbers:
            return f"Under {numbers[0]} INR"

    # Handle "$1000" or numeric values
    numbers = re.findall(r"\d+", b)
    if numbers:
        return f"{numbers[0]} INR"

    # Fallback (return as-is)
    return budget

# -----------------------------
# API Endpoint: Diet Plan
# -----------------------------
@app.post("/diet-plan")
async def generate_diet_plan(request: DietRequest):
    bmi = calculate_bmi(request.height_cm, request.weight_kg)
    normalized_budget = normalize_budget(request.budget)

    user_prompt = (
        f"Create a 1-day healthy meal plan for a person with the following details:\n"
        f"- Age: {request.age}\n"
        f"- Height: {request.height_cm} cm\n"
        f"- Weight: {request.weight_kg} kg\n"
        f"- BMI: {bmi}\n"
        f"- Goal: {request.diet_goal}\n"
        f"- Allergies: {request.allergies or 'None'}\n"
        f"- Daily Budget: {normalized_budget}\n\n"
        f"Constraints:\n"
        f"- Use only Indian cuisine food items.\n"
        f"- Keep total response under 200 words.\n"
        f"- Suggest meals that fit within the given budget.\n"
        f"- Respond ONLY in tabular format with these columns:\n\n"
        f"Meal | Items | Calories (approx) | Cost (INR approx)\n"
        f"-----|--------|------------------|------------------"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a certified dietician and nutrition expert. "
                "Your job is to give meal plans that are clear, realistic, and calorie-balanced."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            stream=False
        )
        final_answer = response.choices[0].message.content
        clean_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
        return {
            "bmi": bmi,
            "budget": normalized_budget,
            "response": clean_answer
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Health check route (optional)
# -----------------------------
@app.get("/")
def root():
    return {"message": "Diet Plan API is running!"}
