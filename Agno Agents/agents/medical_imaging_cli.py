#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    """Get Google API key from environment or prompt user"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API Key: ")
    return api_key

def create_medical_agent(api_key):
    """Create and return the medical agent with the provided API key"""
    return Agent(
        model=Gemini(
            id="gemini-2.0-flash",
            api_key=api_key
        ),
        tools=[DuckDuckGoTools()],
        markdown=True
    )

def analyze_image(image_path, api_key=None):
    """Analyze the medical image using the Gemini model"""
    if not api_key:
        api_key = get_api_key()
        if not api_key:
            print("ERROR: Google API key is required.")
            sys.exit(1)
    
    try:
        # Create the medical agent
        medical_agent = create_medical_agent(api_key)
        
        # Open and resize the image
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))
        
        # Save resized image temporarily
        temp_path = "temp_resized_image.png"
        resized_image.save(temp_path)
        
        # Create AgnoImage object
        agno_image = AgnoImage(filepath=temp_path)
        
        # Medical Analysis Query
        query = """
        You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

        ### 1. Image Type & Region
        - Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
        - Identify the patient's anatomical region and positioning
        - Comment on image quality and technical adequacy

        ### 2. Key Findings
        - List primary observations systematically
        - Note any abnormalities in the patient's imaging with precise descriptions
        - Include measurements and densities where relevant
        - Describe location, size, shape, and characteristics
        - Rate severity: Normal/Mild/Moderate/Severe

        ### 3. Diagnostic Assessment
        - Provide primary diagnosis with confidence level
        - List differential diagnoses in order of likelihood
        - Support each diagnosis with observed evidence from the patient's imaging
        - Note any critical or urgent findings

        ### 4. Patient-Friendly Explanation
        - Explain the findings in simple, clear language that the patient can understand
        - Avoid medical jargon or provide clear definitions
        - Include visual analogies if helpful
        - Address common patient concerns related to these findings

        ### 5. Research Context
        IMPORTANT: Use the DuckDuckGo search tool to:
        - Find recent medical literature about similar cases
        - Search for standard treatment protocols
        - Provide a list of relevant medical links of them too
        - Research any relevant technological advances
        - Include 2-3 key references to support your analysis

        Format your response using clear markdown headers and bullet points. Be concise yet thorough.
        """
        
        # Run analysis
        print("🔄 Analyzing image... Please wait.")
        response = medical_agent.run(query, images=[agno_image])
        
        # Print analysis results
        print("\n📋 ANALYSIS RESULTS")
        print("=" * 80)
        print(response.content)
        print("=" * 80)
        print("\nNote: This analysis is generated by AI and should be reviewed by a qualified healthcare professional.")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"Analysis error: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the program"""
    parser = argparse.ArgumentParser(description="Medical Imaging Diagnosis CLI Tool")
    parser.add_argument("image_path", help="Path to the medical image for analysis")
    parser.add_argument("--api-key", help="Google API key (will use GOOGLE_API_KEY env var if not provided)")
    
    args = parser.parse_args()
    
    print("\n🏥 MEDICAL IMAGING DIAGNOSIS TOOL")
    print("\nDISCLAIMER: This tool is for educational and informational purposes only.")
    print("All analyses should be reviewed by qualified healthcare professionals.")
    print("Do not make medical decisions based solely on this analysis.\n")
    
    analyze_image(args.image_path, args.api_key)

if __name__ == "__main__":
    main()