from google import genai

client = genai.Client(api_key="AIzaSyBRbh0JTGUMSOgZ2QRpA056qyrMinkhnsw")

def fetch_response(query):
    response = client.models.generate_content(
    model="gemini-2.0-flash", contents=query)

    # sanitize response
    response = response.text
    return response
