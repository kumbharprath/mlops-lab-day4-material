from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

demo_app = FastAPI(title="MLOps API Demo")

# Pydantic model for request validation
class Student(BaseModel):
    name: str
    age: int
    course: str


# Temporary in-memory database
students = {}


# GET - Health check
@demo_app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# GET - Fetch all students
@demo_app.get("/students")
def get_students():
    return students


# GET - Fetch student by ID
@demo_app.get("/students/{student_id}")
def get_student(student_id: int):
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    return students[student_id]


# POST - Create a new student
@demo_app.post("/students")
def create_student(student_id: int, student: Student):
    if student_id in students:
        raise HTTPException(status_code=409, detail="Student already exists")

    students[student_id] = student.model_dump()

    return {
        "message": "Student created successfully",
        "student": students[student_id]
    }


# PUT - Update an existing student
@demo_app.put("/students/{student_id}")
def update_student(student_id: int, student: Student):
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    students[student_id] = student.model_dump()

    return {
        "message": "Student updated successfully",
        "student": students[student_id]
    }


# DELETE - Delete a student
@demo_app.delete("/students/{student_id}")
def delete_student(student_id: int):
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    del students[student_id]

    return {"message": "Student deleted successfully"}