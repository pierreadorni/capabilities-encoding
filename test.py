from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os

env = os.environ
api_key = env.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
model_id = "gemini-2.0-flash"

google_search_tool = Tool(google_search=GoogleSearch())

response = client.models.generate_content(
    model=model_id,
    contents="""You are tasked with finding the link to the pdf version of the article introducing a specific remote sensing dataset or model. Search the web for the article and provide the link to the pdf. The dataset or model you are looking for is: 'pspspsps meow meow'.
    your answer needs to contain the following:
    - whether you found the pdf or not
    - the url of the pdf if found
    - any additional information about the search process, such as where the pdf was found or why it was not found""",
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    ),
)

for each in response.candidates[0].content.parts:
    print(each.text)
# Example response:
# The next total solar eclipse visible in the contiguous United States will be on ...

# To get grounding metadata as web content.
print(response.candidates[0].grounding_metadata)
