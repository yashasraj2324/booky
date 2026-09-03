import sys
import os
from pathlib import Path

# Add the project root to sys.path to ensure 'app' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.api.routes.components.prasing.prasing import run

def main():
    doc_path = r"C:\Users\admin\Downloads\Lecturenotes_ML-Module2_P.Ramkumar.docx"
    output_dir = "parsed_doc_output"
    print(f"Testing DOCX parsing for: {doc_path}")
    
    import asyncio
    try:
        asyncio.run(run(doc_path, output_dir))
        print("Parsing successful!")
    except Exception as e:
        print(f"Error during parsing: {e}")

if __name__ == "__main__":
    main()
