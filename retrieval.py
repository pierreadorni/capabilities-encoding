from typing import List
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import io
import httpx
import os
import json
import argparse
import yaspin
from pprint import pprint
import tabulate
import click
import requests

env = os.environ
api_key = env.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

key_hints = {
    "gsd": "[list of numbers] (ground sample distance in meters, ex: [0.5] or [10, 20, 30]. Can be inferred from the sensor if found)",
    "n_samples": "[integer] (number of images in the dataset. if the training and test splits are explicitly separated, use the training set)",
    "task_type": "[string] (classification, segmentation, change detection, object detection, oriented object detection, or super-resolution)",
    "sensor": "[string] (name of the sensor used to capture the images, ex: 'Landsat 8', 'Sentinel-2', 'Aerial')",
}

model_id = "gemini-2.0-flash"
base_prompt_extract = """
This document is an article describing a remote sensing dataset or model. your goal is to find some information in the document and output it in json format. 
The required keys are the following: 
"""
base_prompt_search = """
You are tasked with finding the link to the pdf version of the article introducing a specific remote sensing dataset or model.
Search the web for the article and provide the link to the pdf. your answer needs to contain:
    - whether you found the pdf or not
    - the url of the pdf if found
    - any additional information about the search process, such as where the pdf was found or why it was not found
"""

serialization_prompt = """
You will be given the output of an LLM query, and you need to convert it to a JSON object by extracting the information it contains. your answer needs to be a valid JSON object with the following keys:
    - found (boolean): whether the LLM found the pdf or not
    - url (str optional): the url of the pdf if found
    - explanation (str optional): any additional information about the search process, such as where the pdf was found or why it was not found

Make sure to check the output of the LLM query and extract the relevant information. If the LLM did not find the precise PDF the user wanted (for example, if it found some related work but not the OG), you should output found:false and not give a URL. 
"""

google_search_tool = Tool(google_search=GoogleSearch())


def make_prompt_extract(base_prompt, keys):
    prompt = base_prompt
    for key in keys:
        prompt += f"\n- {key}"
        if key in key_hints:
            prompt += f" {key_hints[key]}"
    return prompt


def make_prompt_search(base_prompt, dataset_name):
    prompt = base_prompt
    prompt += f"\nThe dataset or model you are looking for is: '{dataset_name}'."
    return prompt


def make_prompt_serialization(search_query, search_answer):
    prompt = serialization_prompt
    prompt += f"\n\nThe user originally asked: {search_query}"
    prompt += f"\n\nThe LLM response is: {search_answer}"
    return prompt


def process_pdf(url: str, keys: List[str]) -> dict:
    with yaspin.yaspin() as sp:
        sp.text = f"Resolving redirects for {url}"
        var_output_url = requests.head(url, allow_redirects=True)
        url = var_output_url.url
        sp.text = f"Downloading {url}"
        doc_io = io.BytesIO(httpx.get(url).content)
        doc = client.files.upload(file=doc_io, config={"mime_type": "application/pdf"})
        sp.text = "Processing PDF"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[doc, make_prompt_extract(base_prompt_extract, keys)],
            config={"response_mime_type": "application/json", "temperature": 0},
        )
        sp.ok("✅ ")

    return json.loads(response.text)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("url", type=str)
@click.option(
    "--keys",
    type=str,
    multiple=True,
    help="List of keys to extract from the PDF",
    default=["gsd", "n_samples", "task_type"],
)
def extract(url, keys):
    result = process_pdf(url, keys)
    if type(result) == list:
        result = result[0]
    table = list(result.items())
    print(tabulate.tabulate(table, tablefmt="fancy_grid"))


def search_pdf(dataset_or_model_name):
    response = client.models.generate_content(
        model=model_id,
        contents=make_prompt_search(base_prompt_search, dataset_or_model_name),
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        ),
    )

    response_json = client.models.generate_content(
        model=model_id,
        contents=make_prompt_serialization(dataset_or_model_name, response.text),
        config=GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # To get grounding metadata as web content.
    resp = response_json.text.strip("` ")
    result = json.loads(resp)
    if type(result) == list:
        result = result[0]

    return result


@cli.command()
@click.argument("dataset_name", type=str)
def search(dataset_name):
    result = search_pdf(dataset_name)
    table = list(result.items())
    print(tabulate.tabulate(table, tablefmt="fancy_grid"))


@cli.command()
@click.argument("dataset_name", type=str)
@click.option(
    "--keys",
    type=str,
    multiple=True,
    help="List of keys to extract from the PDF",
    default=["gsd", "n_samples", "task_type"],
)
def search_and_extract(dataset_name, keys):
    with yaspin.yaspin() as sp:
        sp.text = f"Searching for {dataset_name}"
        result = search_pdf(dataset_name)
    if result["found"]:
        print(f"PDF found at {result['url']}")
        result = process_pdf(result["url"], keys)
        if type(result) == list:
            result = result[0]
        table = list(result.items())
        print(tabulate.tabulate(table, tablefmt="fancy_grid"))
    else:
        print("PDF not found")


if __name__ == "__main__":
    cli()
