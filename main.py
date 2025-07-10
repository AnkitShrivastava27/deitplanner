from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for any frontend (like Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a better free model for instruction-based generation
model_id = "google/flan-t5-base"
client = InferenceClient(model=model_id)

# Request schema from frontend
class DietRequest(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    diet_goal: str  # Example: "weight loss", "muscle gain"
    allergies: str = ""  # Optional

# BMI calculation function
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

# API endpoint
@app.post("/diet-plan")
async def generate_diet_plan(request: DietRequest):
    bmi = calculate_bmi(request.height_cm, request.weight_kg)

    prompt = (
        f"Generate a healthy one-day meal plan for a {request.age}-year-old "
        f"person who is {request.height_cm} cm tall, weighs {request.weight_kg} kg, "
        f"has a BMI of {bmi}, goal is {request.diet_goal}, and allergies: {request.allergies or 'none'}.\n"
        f"Please list meals: Breakfast, Lunch, Snack, Dinner with approximate calories."
    )

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=256
        )
        return {
            "bmi": bmi,
            "response": response.strip()
        }
    except Exception as e:
        return {"error": str(e)}
