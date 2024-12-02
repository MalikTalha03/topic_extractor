from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import openai
from app import process_assignment, get_results, download_pdf

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Topic Extractor API!"}

from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
# Text-based endpoint
@app.post("/process-url/")
async def process_text(input: TextInput):
    try:
        text = input.text
        file_location = download_pdf(text, "temp.pdf")
        # Process the file
        ranked_results = process_assignment(file_location)

        # Remove the temporary file
        os.remove(file_location)

        return JSONResponse(content={"ranked_results": ranked_results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


