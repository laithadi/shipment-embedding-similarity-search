import logging
import torch

# setup logger
logger = logging.getLogger(__name__)


class HuggingFaceEmbedding:
    """
    Base class for HuggingFace embedding models.

    This class initializes the tokenizer and model and provides functionality
    to encode text into embeddings.

    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer for tokenizing input text.
        model (PreTrainedModel): The HuggingFace model used for generating embeddings.
        return_tensors (str): The format of the output tensors (default is "pt" for PyTorch).
        truncate (bool): Whether to truncate input text that exceeds the maximum length.
        padding (bool): Whether to pad the input text to the maximum length.
        emb_max_len (int): The maximum length for input text.
    """

    def __init__(
        self, 
        tokenizer, 
        model, 
        return_tensors: str = "pt",
        truncate: bool = True, 
        padding: bool = True,
        emb_max_len: int = 100,
    ):
        """
        Initializes the HuggingFaceEmbedding class with tokenizer, model, and other configuration options.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing input text.
            model (PreTrainedModel): The HuggingFace model to be used for generating embeddings.
            return_tensors (str, optional): The format of the output tensors (default is "pt" for PyTorch).
            truncate (bool, optional): Whether to truncate input text that exceeds the maximum length (default is True).
            padding (bool, optional): Whether to pad the input text to the maximum length (default is True).
            emb_max_len (int, optional): The maximum length for input text (default is 100).
        """
        self.tokenizer = tokenizer
        self.model = model
        self.return_tensors = return_tensors
        self.truncate = truncate
        self.padding = padding
        self.emb_max_len = emb_max_len

    def encode(self, text):
        """
        Encodes input text into embeddings using the provided tokenizer and model.

        This method tokenizes the input text, passes it through the model, and retrieves
        the [CLS] token embedding, which represents the entire input text.

        Args:
            text (str): The input text to be encoded.

        Returns:
            numpy.ndarray: The embedding of the input text as a NumPy array.

        Raises:
            Exception: If an error occurs during text embedding.
        """
        try:
            # tokenize the input text
            inputs = self.tokenizer(
                text, 
                return_tensors=self.return_tensors, 
                truncation=self.truncate, 
                padding=self.padding, 
                max_length=self.emb_max_len,
            )

            # get embeddings from the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # return the [CLS] token embedding (first token)
            # outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
            return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        except Exception as eee:
            # log the error and re-raise it
            logger.error(f"Error embedding text: {str(eee)}")
            raise eee
