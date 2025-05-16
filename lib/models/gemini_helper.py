from google import genai

client = genai.Client(api_key="YOUR API KEY")

def fetch_response(query):
    response = client.models.generate_content(
    model="gemini-2.0-flash", contents=query)

    # sanitize response
    response = response.text
    return response

    
    
