from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import openai
from main import process_assignment, get_results

app = FastAPI()

# File-based endpoint
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Process the file
        ranked_results = process_assignment(file_location)

        # Remove the temporary file
        os.remove(file_location)

        return JSONResponse(content={"ranked_results": ranked_results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
# Text-based endpoint
@app.post("/process-text/")
async def process_text(input: TextInput):
    try:
        text = input.text
        ranked_results = get_results(text)

        return JSONResponse(content={"ranked_results": ranked_results[:5]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
