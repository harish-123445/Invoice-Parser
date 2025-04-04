# Invoice OCR API

A Flask-based API that processes PDF invoices to extract structured data using OCR and Google's Gemini API.

## Setup

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/harish-123445/Invoice-Parser.git
    cd invoice-ocr-api
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:

    Create a `.env` file in the project root with the following content:

    ```ini
    GEMINI_API_KEY=your_gemini_api_key
    ```

    Replace `your_gemini_api_key` with your actual Google Gemini API key.

## Running the Application

Start the Flask application:

```bash
python app.py
```

By default, the API will be accessible at `http://127.0.0.1:5000`.

## Usage

To test the API, use the following `curl` command to send a PDF invoice for processing:

```bash
curl -X POST http://127.0.0.1:5000/process-invoice \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_your_invoice.pdf"
```

Replace `path_to_your_invoice.pdf` with the path to the PDF invoice you want to process.

## Health Check

Verify the API is running with:

```bash
curl http://127.0.0.1:5000/health
```

A successful response will indicate the API is operational. 
