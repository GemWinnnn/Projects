from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import json
import re
import tempfile
from dotenv import load_dotenv, dotenv_values
from pydantic import BaseModel
from typing import Optional

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
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
            
            # Enhanced prompt to ensure clean JSON output
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

            Return the response as a valid JSON object with these exact field names: 
            "patient_name", "doctor_name", "date", "medication_name", "dosage", "schedule", 
            "directions", "quantity", "refills", "pharmacy_info". Use null for any field not found. 
            Ensure the response contains ONLY the JSON object, with no additional text, code blocks, or comments. 
            Example:
            {
                "patient_name": "John Doe",
                "doctor_name": "Dr. Smith",
                "date": "2025-01-01",
                "medication_name": "Amoxicillin",
                "dosage": "500 mg",
                "schedule": "Twice daily",
                "directions": "Take with food",
                "quantity": "30 capsules",
                "refills": "0",
                "pharmacy_info": null
            }
            """
            
            # Send the image to Gemini for analysis
            response = self.model.generate_content([prompt, img])
            
            # Check if response is valid
            if not response.text:
                return {"error": "Empty response from Gemini API"}
            
            response_text = response.text
            print("Raw response:", response_text)  # Debug: Log raw response
            
            # Try to extract JSON using regex
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group().strip()
            else:
                print("No JSON found, using fallback parsing")  # Debug
                return self._extract_from_text(response_text)
            
            print("Extracted json_str:", json_str)  # Debug: Log extracted JSON
            
            # Parse the JSON
            try:
                prescription_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print("JSON parsing error:", str(e))  # Debug: Log parsing error
                return self._extract_from_text(response_text)
            
            return prescription_data
            
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    
    def _extract_from_text(self, text):
        """
        Fallback method to extract prescription data from free text if JSON parsing fails.
        
        Args:
            text: The text response from Gemini
            
        Returns:
            Dictionary with extracted prescription information
        """
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
        
        lines = text.split('\n')
        field_map = {
            "patient_name": ["patient name", "patient:", "name:"],
            "doctor_name": ["doctor", "physician", "prescriber", "doctor name"],
            "date": ["date"],
            "medication_name": ["medication", "drug", "medicine"],
            "dosage": ["dosage", "strength"],
            "schedule": ["schedule", "frequency"],
            "directions": ["direction", "instruction", "sig"],
            "quantity": ["quantity", "amount"],
            "refills": ["refill"],
            "pharmacy_info": ["pharmacy"]
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for field, keywords in field_map.items():
                if any(keyword in line.lower() for keyword in keywords):
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    prescription_data[field] = value
                    break
        
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
    # Try to get API key from environment variable
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
        
        # Initialize the prescription reader
        reader = PrescriptionReader(api_key)
        
        # Extract information from the prescription
        prescription_info = reader.extract_prescription_info(filepath)
        
        # Clean up - remove the temporary file
        os.remove(filepath)
        
        # Check for errors
        if "error" in prescription_info:
            return jsonify(prescription_info), 500
            
        # Validate the prescription data
        validation_result = reader.validate_prescription(prescription_info)
        prescription_info["validation"] = validation_result["flags"]
        
        return jsonify(prescription_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)