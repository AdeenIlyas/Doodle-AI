from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
from together import Together
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)
CORS(app)

client = Together(api_key=api_key)

def encode_image(image):
    """Encodes a PIL image as base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_to_text(client, model, base64_image, prompt):
    """Uses the Together API to generate text from the provided image."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=model
    )
    return chat_completion.choices[0].message.content

@app.route('/describe_image', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data or not isinstance(image_data, str) or ',' not in image_data:
            return jsonify({"error": "Invalid image data format."}), 400

        # Decode the image from base64
        image_data = image_data.split(',')[1] 
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes))

        # Resizing the image
        target_size = (256, 256)
        image = image.resize(target_size, Image.LANCZOS)

        # encode image to parse as URL
        base64_image = encode_image(image)
        
        prompt = "Describe the image"
        
        # get response
        return jsonify({'prediction': image_to_text(client, model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", base64_image=base64_image, prompt=prompt)}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process the image. {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
