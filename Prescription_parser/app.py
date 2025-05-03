from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import json
import tempfile
from dotenv import load_dotenv, dotenv_values
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()
config = dotenv_values(".env")

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file size

# Pydantic model for prescription data
class PrescriptionData(BaseModel):
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    medication_name: Optional[str] = None
    dosage: Optional[str] = None
    schedule: Optional[str] = None
    directions: Optional[str] = None
    quantity: Optional[str] = None
    refills: Optional[str] = None
    pharmacy_info: Optional[str] = None

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class PrescriptionReader:
    """
    A class to extract prescription information from images using the Gemini API.
    """
    
    def __init__(self, api_key):
        """
        Initialize the PrescriptionReader with your Gemini API key.
        
        Args:
            api_key: Your Google Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Pro Vision for image analysis
        self.model = genai.GenerativeModel('gemini-pro-vision')
        
    def extract_prescription_info(self, image_path):
        """
        Extract prescription information from an image.
        
        Args:
            image_path: Path to the prescription image file
            
        Returns:
            Dictionary containing structured prescription information
        """
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Create a structured prompt to extract specific information
            prompt = """
            You are a medical prescription analyzer. Extract the following information from this prescription image:
            
            1. Patient Name
            2. Doctor Name
            3. Date
            4. Medication Name
            5. Dosage (strength)
            6. Schedule (how often to take)
            7. Directions (how to take)
            8. Quantity
            9. Refills (if mentioned)
            10. Pharmacy Info (if available)
            
            Format your response as a clean JSON object with these fields. If any field is not found in the image, set its value to null.
            """
            
            # Send the image to Gemini for analysis
            response = self.model.generate_content([prompt, img])
            
            # Extract and parse the JSON response
            response_text = response.text
            
            # Try to extract JSON from the response
            # First, look for JSON in code blocks
            if "```json" in response_text and "```" in response_text.split("```json")[1]:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text and "```" in response_text.split("```")[1]:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                # If no code blocks, try to find a JSON-like structure
                json_str = response_text
            
            # Parse the JSON
            try:
                prescription_data = json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, create a structured response with what we can extract
                prescription_data = self._extract_from_text(response_text)
            
            return prescription_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_from_text(self, text):
        """
        Fallback method to extract prescription data from free text if JSON parsing fails.
        
        Args:
            text: The text response from Gemini
            
        Returns:
            Dictionary with extracted prescription information
        """
        # Initialize the prescription data structure
        prescription_data = {
            "patient_name": None,
            "doctor_name": None,
            "date": None,
            "medication_name": None,
            "dosage": None,
            "schedule": None,
            "directions": None,
            "quantity": None,
            "refills": None,
            "pharmacy_info": None
        }
        
        # Simple text extraction based on common patterns
        lines = text.split('\n')
        current_field = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for field indicators
            if "patient name" in line.lower() or "patient:" in line.lower():
                current_field = "patient_name"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
            
            elif "doctor" in line.lower() or "physician" in line.lower() or "prescriber" in line.lower():
                current_field = "doctor_name"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
            
            elif "date" in line.lower():
                current_field = "date"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "medication" in line.lower() or "drug" in line.lower():
                current_field = "medication_name"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "dosage" in line.lower() or "strength" in line.lower():
                current_field = "dosage"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "schedule" in line.lower() or "frequency" in line.lower():
                current_field = "schedule"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "direction" in line.lower() or "instruction" in line.lower() or "sig" in line.lower():
                current_field = "directions"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "quantity" in line.lower() or "amount" in line.lower():
                current_field = "quantity"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "refill" in line.lower():
                current_field = "refills"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            elif "pharmacy" in line.lower():
                current_field = "pharmacy_info"
                value = line.split(":", 1)[1].strip() if ":" in line else None
                if value:
                    prescription_data[current_field] = value
                    
            # If we have a current field but no value yet, this line might be the value
            elif current_field and not prescription_data[current_field]:
                prescription_data[current_field] = line
        
        return prescription_data

    def validate_prescription(self, prescription_data):
        """
        Validate the extracted prescription data and flag any potential issues.
        
        Args:
            prescription_data: Dictionary containing the extracted prescription information
            
        Returns:
            Dictionary with the original data and any validation flags
        """
        validation = {"data": prescription_data, "flags": []}
        
        # Check for missing critical information
        if not prescription_data.get("medication_name"):
            validation["flags"].append("MISSING_MEDICATION")
            
        if not prescription_data.get("dosage"):
            validation["flags"].append("MISSING_DOSAGE")
            
        if not prescription_data.get("directions"):
            validation["flags"].append("MISSING_DIRECTIONS")
            
        if not prescription_data.get("doctor_name"):
            validation["flags"].append("MISSING_DOCTOR")
        
        return validation

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-prescription', methods=['POST'])
def analyze_prescription():
    # Try to get API key from request or use environment variable
    api_key = request.form.get('api_key')
    if not api_key:
        api_key = config.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({"error": "No Gemini API key provided"}), 400
    
    # Check if a file was uploaded
    if 'prescription_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['prescription_image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Please upload a JPG or PNG image."}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize the prescription reader with the provided API key
        reader = PrescriptionReader(api_key)
        
        # Extract information from the prescription
        prescription_info = reader.extract_prescription_info(filepath)
        
        # Validate the prescription data
        validation_result = reader.validate_prescription(prescription_info)
        
        # Add validation results to the response
        prescription_info["validation"] = validation_result["flags"]
        
        # Clean up - remove the temporary file
        os.remove(filepath)
        
        # Return the extracted information
        return jsonify(prescription_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)