import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import fitz
from PIL import Image
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFInvoiceOCRParser:
    """
    A class to parse PDF invoices using OCR and extract structured data with Google's Gemini API.
    Also tracks token usage from the API.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the PDF parser.
        
        Args:
            gemini_api_key: Google Gemini API key. If None, will try to read from environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it to the constructor.")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the Gemini Vision model for image processing
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        logger.info("PDFInvoiceOCRParser initialized successfully")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> list[Image.Image]:
        """
        Convert each page of a PDF file to an image.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for the converted images
            
        Returns:
            List of PIL Image objects
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Convert page to a pixmap (image)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(img)
            
            logger.debug(f"Converted page {page_num + 1} to image")
        
        logger.info(f"PDF conversion complete. Generated {len(images)} images.")
        return images
    
    def extract_invoice_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract structured data from an invoice image using Google's Gemini API.
        Also tracks token usage from the API.
        
        Args:
            image: PIL Image object of the invoice
            
        Returns:
            Dictionary with extracted key-value pairs from the invoice
        """
        logger.info("Extracting invoice data from image")
        
        # Prepare the prompt for Gemini
        prompt = """
        Please analyze the provided invoice image using OCR technology and extract the following information in a structured JSON format:
        Required Fields

        -Invoice Number (also extract ACK NO if present, as Invoice Number)
        -Invoice Date
        -Due Date
        -Vendor Name
        -Vendor Address
        -Customer/Client Name
        -Customer/Client Address
        -Line Items (including quantity, description, unit price, tax price, and total price)
        -Subtotal
        -Tax Amount
        -Total Amount Due
        -Payment Terms
        -Payment Method (if available)
        -Sales Order Number (if available)
        -Buyer Order Number (if available)
        -Purchase Order Number (also labeled as PO Number or Buyer Order Number)

        Extract all PO numbers, whether digital or handwritten. If multiple PO numbers exist, return them as a list
        -Invoice URL (if available)
        Special Instructions

        -The invoice may contain both digital and handwritten text - extract both
        -For fields with both digital and handwritten versions (like PO Numbers), combine all instances into a single list
        -Return results in properly formatted JSON
        -For any field not found in the image, set the value to null
        -Don't map the same value to multiple fields
        -If a field is not applicable, set it to null
        Please ensure all relevant information is accurately extracted, regardless of format or placement within the invoice.
    """
        
        try:
            # Call Gemini API with the image
            response = self.model.generate_content([prompt, image])
            
            # Track token usage if available in response
            try:
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        self.total_input_tokens += response.usage_metadata.prompt_token_count
                    elif hasattr(response.usage_metadata, 'prompt_tokens'):
                        self.total_input_tokens += response.usage_metadata.prompt_tokens
                    
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        self.total_output_tokens += response.usage_metadata.candidates_token_count
                    elif hasattr(response.usage_metadata, 'completion_tokens'):
                        self.total_output_tokens += response.usage_metadata.completion_tokens
                    elif hasattr(response.usage_metadata, 'response_tokens'):
                        self.total_output_tokens += response.usage_metadata.response_tokens
                        
                # Try alternate response structure if available
                elif hasattr(response, 'tokens'):
                    if hasattr(response.tokens, 'input'):
                        self.total_input_tokens += response.tokens.input
                    if hasattr(response.tokens, 'output'):
                        self.total_output_tokens += response.tokens.output
                
                # Increment pages processed
                self.pages_processed += 1
                
            except Exception as token_err:
                logger.warning(f"Could not track token usage: {token_err}")
            
            # Extract JSON from the response
            response_text = response.text
            
            # Find JSON content in the response (in case it's wrapped in markdown code blocks)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                try:
                    # Parse the JSON data
                    invoice_data = json.loads(json_content)
                    logger.info("Successfully extracted and parsed invoice data")
                    return invoice_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Gemini response: {e}")
                    return {"error": "Failed to parse JSON from API response", "raw_response": response_text}
            else:
                logger.warning("No JSON content found in Gemini response")
                return {"error": "No JSON content found in API response", "raw_response": response_text}
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {"error": f"API call failed: {str(e)}"}
    
    def process_invoice(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF invoice: convert to images and extract data from each page.
        
        Args:
            pdf_path: Path to the PDF invoice file
            
        Returns:
            Dictionary containing extracted data from all pages and token usage info
        """
        logger.info(f"Processing invoice: {pdf_path}")
        
        # Reset token counters for this processing run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {"status": "error", "message": f"PDF file not found: {pdf_path}"}
        
        try:
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)
            
            # Process each image
            results = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} of {len(images)}")
                page_data = self.extract_invoice_data(image)
                
                # Add page information
                page_data["page_number"] = i + 1
                page_data["total_pages"] = len(images)
                
                results.append(page_data)
            
            # Create token usage summary
            token_summary = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "pages_processed": self.pages_processed,
                "tokens_per_page": {
                    "input": self.total_input_tokens / max(1, self.pages_processed),
                    "output": self.total_output_tokens / max(1, self.pages_processed),
                    "total": (self.total_input_tokens + self.total_output_tokens) / max(1, self.pages_processed)
                }
            }
            
            logger.info(f"Invoice processing complete. Extracted data from {len(results)} pages.")
            
            return {
                "status": "success", 
                "data": results, 
                "token_usage": token_summary
            }
            
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            return {"status": "error", "message": f"Failed to process invoice: {str(e)}"}

# Initialize Flask app
app = Flask(__name__)

# Create a parser instance - initialize at the module level
parser = None

# Create a function to initialize the parser
def get_parser():
    global parser
    if parser is None:
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY environment variable not set")
            parser = PDFInvoiceOCRParser(gemini_api_key=api_key)
        except Exception as e:
            logger.critical(f"Failed to initialize parser: {e}")
            raise
    return parser

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "message": "Invoice OCR API is operational"})

@app.route('/process-invoice', methods=['POST'])
def process_invoice_endpoint():
    """
    API endpoint to process PDF invoices.
    
    Expects a multipart/form-data request with a PDF file.
    Returns extracted invoice data in JSON format.
    """
    try:
        # Get or initialize the parser
        try:
            current_parser = get_parser()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"API initialization failed: {str(e)}"
            }), 500
        
        # Check if request has a file
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({
                "status": "error", 
                "message": "No file provided. Please upload a PDF file."
            }), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({
                "status": "error", 
                "message": "No file selected. Please select a PDF file."
            }), 400
        
        # Check file extension
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                "status": "error", 
                "message": "Only PDF files are supported."
            }), 400
        
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Process the invoice
        logger.info(f"Processing uploaded invoice: {file.filename}")
        result = current_parser.process_invoice(temp_path)
        
        # Remove the temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")
        
        # Return the results
        return jsonify(result), 200 if result.get("status") == "success" else 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

# Create a simple auth middleware (optional)
@app.before_request
def check_api_key():
    """
    Basic API key authentication middleware.
    
    This is an optional security measure. You can customize or remove this
    based on your security requirements.
    """
    # Skip auth for health endpoint
    if request.path == '/health':
        return None
    
    # Get API key from request header or query param
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    
    # Check if API key is required and valid
    if os.environ.get('REQUIRE_API_KEY', 'false').lower() == 'true':
        valid_api_key = os.environ.get('API_KEY')
        if not api_key or api_key != valid_api_key:
            logger.warning("Invalid or missing API key")
            return jsonify({
                "status": "error",
                "message": "Invalid or missing API key"
            }), 401

if __name__ == '__main__':
    # Initialize parser before starting the server
    try:
        parser = get_parser()
        logger.info("Parser initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize parser: {e}")
        raise
    
    # Start the Flask server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)