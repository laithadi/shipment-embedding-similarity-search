import logging
import numpy as np
import torch
import unittest

from transformers import DistilBertModel, DistilBertTokenizer

from .base_huggingface_embedding import HuggingFaceEmbedding

# setup logger 
logger = logging.getLogger(__name__)


class DistilBertTextEmbedding(HuggingFaceEmbedding):
    """
    DistilBERT-based embedding for textual columns.
    
    Inherits from HuggingFaceEmbedding and initializes the DistilBERT model and tokenizer.
    This class provides the ability to embed text using the DistilBERT model.
    """

    def __init__(self):
        """
        Initializes the DistilBERT tokenizer and model.
        
        Loads the pretrained DistilBERT model and tokenizer for text embedding.
        """
        logger.info("Creating the DistilBert tokenizer and model.")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Initialize parent class with tokenizer and model
        super().__init__(tokenizer, model)

    def embed_text(self, text):
        """
        Embed text using the DistilBERT model.
        
        Args:
            text (str): The input text to embed.
        
        Returns:
            numpy.ndarray: The embedded vector representation of the input text.
        """
        try: 
            logger.info(f"Encoding {text}...")
            return self.encode(text)

        except Exception as eee:
            logger.error(f"Error embedding text: {str(eee)}")
            raise eee


class TestDistilBertTextEmbedding(unittest.TestCase):
    """
    Unit test class for testing the DistilBertTextEmbedding class.
    
    Contains tests to validate the embedding functionality of the DistilBertTextEmbedding class.
    """

    distilbert_embed = None 

    @classmethod
    def setUpClass(self):
        """
        Set up the test environment.
        
        Initializes the DistilBertTextEmbedding class for use in the tests.
        """
        self.distilbert_embed = DistilBertTextEmbedding()

    def test_embed_text(self):
        """
        Test the embedding of text.
        
        Verifies that the embedded text is returned as a numpy ndarray.
        """
        txt_to_embed = """The sun was setting over the distant hills, casting a warm, golden hue across the fields. Birds flitted between the branches of the trees, their songs blending with the soft rustle of leaves in the breeze. The air smelled of fresh earth and wildflowers, and everything seemed to slow down, as if the world was taking a deep breath before the night arrived. In the distance, a small river meandered through the valley, its surface catching the last rays of sunlight. The quiet murmurs of the water combined with the sounds of nature created a peaceful symphony, one that made time feel less important. Nearby, a family was gathering around a picnic blanket, laughing and sharing stories of the day. The children ran barefoot through the grass, their energy endless, while the adults sat back and enjoyed the simplicity of the moment. As the first stars began to appear in the sky, the warmth of the day gradually gave way to the coolness of evening. The scene felt timeless, as though it had been this way for centuries. The combination of nature, family, and the setting sun created an atmosphere of perfect serenity, a peaceful refuge from the bustle of the outside world."""

        embedded_vector = self.distilbert_embed.embed_text(text= txt_to_embed)

        self.assertIsInstance(embedded_vector, np.ndarray)
