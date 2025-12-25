from typing import Tuple, Callable
from pydantic import BaseModel, Field

print("--- 1. As a 'pass' replacement ---")
def future_function():
    ...  # valid syntax, does nothing just like 'pass'

future_function()
print("Function executed successfully.")


print("\n--- 2. In Type Hints ---")
# Tuple[int, ...] means a tuple of ANY number of integers
def process_numbers(numbers: Tuple[int, ...]):
    print(f"Received tuple with {len(numbers)} items: {numbers}")

process_numbers((1, 2, 3, 4, 5))


print("\n--- 3. In Pydantic (FastAPI style) ---")
class User(BaseModel):
    name: str = Field(...)  # The '...' means this field is REQUIRED
    age: int = Field(None)  # This field is OPTIONAL

try:
    print("Attempting to create User without 'name'...")
    User(age=25)
except Exception as e:
    print(f"Caught expected error:\n{e}")

print("\nAttempting to create valid User...")
u = User(name="Alice", age=30)
print(f"Success: {u}")
