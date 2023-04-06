from typing import List
import openai
import os
import cohere

from tenacity import retry, wait_random_exponential, stop_after_attempt


EMBEDDINGS_SERVICE = os.environ.get("EMBEDDINGS_SERVICE", "openai")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    if EMBEDDINGS_SERVICE == "openai":
        # Call the OpenAI API to get the embeddings
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        # Extract the embedding data from the response
        data = response["data"]  # type: ignore
        # Return the embeddings as a list of lists of floats
        return [result["embedding"] for result in data]
    elif EMBEDDINGS_SERVICE == "cohere":
        if COHERE_API_KEY is None:
            raise ValueError(
                "COHERE_API_KEY environment variable must be set to use the cohere embeddings service."
            )
        # Call the Cohere API to get the embeddings
        co = cohere.Client(COHERE_API_KEY)
        response = co.embed(texts=texts, model='multilingual-22-12')
        return response.embeddings


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model="gpt-3.5-turbo",  # use "gpt-4" for better results
):
    """
    Generate a chat completion using OpenAI's chat completion API.

    Args:
        messages: The list of messages in the chat history.
        model: The name of the model to use for the completion. Default is gpt-3.5-turbo, which is a fast, cheap and versatile model. Use gpt-4 for higher quality but slower results.

    Returns:
        A string containing the chat completion.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    # call the OpenAI chat completion API with the given messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    print(f"Completion: {completion}")
    return completion
