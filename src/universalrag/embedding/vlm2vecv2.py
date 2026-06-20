from typing import List, Union, Optional
from PIL import Image
import logging
import os
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .VLM2Vec.src.arguments import ModelArguments, DataArguments
from .VLM2Vec.src.model.model import MMEBModel
from .VLM2Vec.src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS, Qwen2_VL_process_fn
from .VLM2Vec.src.utils import batch_to_device
from .VLM2Vec.src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info


def _load_image(path: str) -> Image.Image:
    """Load an image and fully decode it so the file handle is released.

    PIL's Image.open is lazy and keeps the file descriptor open until the
    pixels are accessed. Opening many images up front (e.g. a whole dataset)
    therefore leaks descriptors and hits the OS limit ([Errno 24] Too many
    open files). Reading inside a `with` block and converting to RGB forces a
    full load, so the descriptor is closed immediately.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


class VLM2VecV2EmbeddingModel:
    """Wrapper for VLM2Vec V2.0 Embedding Model
    
    VLM2Vec V2.0 is a unified multimodal embedding model that can encode
    images, videos, and text into a shared embedding space.
    """
    
    def __init__(self, 
                 model_name: str = "VLM2Vec/VLM2Vec-V2.0",
                 pooling: str = "last",
                 normalize: bool = True,
                 device: str = "cuda"):
        """
        Initialize VLM2Vec V2.0 model
        
        Args:
            model_name: Model name/path for VLM2Vec V2.0
            pooling: Pooling strategy ('last', 'mean', etc.)
            normalize: Whether to normalize embeddings
            device: Device to run model on
        """
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self.device = device
        
        # Model components
        self.model = None
        self.processor = None
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the VLM2Vec V2.0 model and processor"""
        # Set up model arguments
        model_args = ModelArguments(
            model_name=self.model_name,
            pooling=self.pooling,
            normalize=self.normalize,
            model_backbone='qwen2_vl',
            lora=True
        )
        
        # Set up data arguments
        data_args = DataArguments()
        
        # Load processor
        self.processor = load_processor(model_args, data_args)
        
        # Load model
        self.model = MMEBModel.load(model_args, is_trainable=False)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        
        logging.info(f"Successfully loaded VLM2Vec V2.0 model: {self.model_name}")
    
    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode text strings into embeddings
        
        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional arguments
        
        Returns:
            numpy array of embeddings with shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        batch_size = kwargs.get('batch_size', 16)

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text batches"):
                batch_texts = texts[i:i + batch_size]

                # Use Qwen2_VL_process_fn for true batch processing.
                # It handles text-only inputs (images=None) and pads input_ids
                # across the batch, same as the image path.
                processor_inputs = {
                    "text": batch_texts,
                    "images": [None] * len(batch_texts),
                }
                inputs = Qwen2_VL_process_fn(processor_inputs, self.processor)
                inputs = batch_to_device(inputs, self.device)

                output = self.model(tgt=inputs)["tgt_reps"]

                embeddings.append(output.float().cpu().numpy())

        # Stack all embeddings
        result = np.vstack(embeddings)
        return result
    
    def encode_image(self, images: Union[Image.Image, List[Image.Image], str, List[str]], 
                     query_text: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Encode images into embeddings using batch processing
        
        Args:
            images: Single PIL Image, list of PIL Images, single image path (str), or list of image paths (List[str])
            query_text: Optional query text to guide image embedding
            **kwargs: Additional arguments
        
        Returns:
            numpy array of embeddings with shape (num_images, embedding_dim)
        """
        # Convert inputs to list of PIL Images for consistent batch processing
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, str):
            images = [_load_image(images)]
        elif isinstance(images, list) and isinstance(images[0], str):
            max_workers = kwargs.get('max_workers', os.cpu_count())
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                images = list(tqdm(executor.map(_load_image, images), total=len(images), desc="Loading images"))

        # Default query text for image representation
        if query_text is None:
            query_text = f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'
        
        # Get batch size from kwargs, default to 8 for images
        batch_size = kwargs.get('batch_size', 8)
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Encoding image batches"):
                batch_images = images[i:i + batch_size]
                
                # Prepare query texts for this batch
                query_texts = [query_text] * len(batch_images)
                
                # Prepare processor inputs for batch processing
                processor_inputs = {
                    "text": query_texts,
                    "images": batch_images,
                }
                
                # Use Qwen2_VL_process_fn for batch processing
                inputs = Qwen2_VL_process_fn(processor_inputs, self.processor)
                inputs = batch_to_device(inputs, self.device)
                
                with autocast(device_type=self.device.split(':')[0], dtype=torch.bfloat16):
                    output = self.model(qry=inputs)["qry_reps"]
                
                embeddings.append(output.float().cpu().numpy())
        
        # Stack all embeddings
        result = np.vstack(embeddings)
        return result
    
    def encode_video(self, video_paths: Union[str, List[str]], 
                     query_text: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Encode videos into embeddings
        
        Args:
            video_paths: Single video path or list of video paths
            query_text: Optional query text to guide video embedding
            **kwargs: Additional arguments (max_pixels, fps, nframes, etc.)
        
        Returns:
            numpy array of embeddings with shape (num_videos, embedding_dim)
        """
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        embeddings = []
        max_pixels = kwargs.get('max_pixels', 360 * 420)
        nframes = kwargs.get('nframes', 16)
        
        # Default query text for video representation
        if query_text is None:
            query_text = f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.'
        
        with torch.no_grad():
            for video_path in tqdm(video_paths, desc="Encoding videos"):
                # Prepare video message format (following the demo)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": max_pixels,
                                "nframes": nframes,
                            },
                            {"type": "text", "text": "Describe this video."},
                        ],
                    }
                ]
                
                # Process video inputs using VLM2Vec utility
                _, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=query_text,
                    videos=video_inputs,
                    return_tensors="pt"
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # Handle video-specific tensor formatting (required for VLM2Vec V2.0)
                if 'pixel_values_videos' in inputs:
                    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
                if 'video_grid_thw' in inputs:
                    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
                
                # Get video embeddings
                with autocast(device_type=self.device.split(':')[0], dtype=torch.bfloat16):
                    output = self.model(qry=inputs)["qry_reps"]
                
                embeddings.append(output.float().cpu().numpy())
        
        # Stack all embeddings
        result = np.vstack(embeddings)
        return result
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
        
        Returns:
            Similarity scores
        """
        # Convert to tensors
        emb1 = torch.from_numpy(embeddings1).to(self.device)
        emb2 = torch.from_numpy(embeddings2).to(self.device)
        
        # Use model's similarity computation
        with torch.no_grad():
            similarity = self.model.compute_similarity(emb1, emb2)
        
        return similarity.cpu().numpy()
    
    def encode(self, content: Union[str, List[str], Image.Image, List[Image.Image]], 
               modality: str = "text", **kwargs) -> np.ndarray:
        """
        Universal encode method that routes to appropriate encoder based on modality
        
        Args:
            content: Content to encode (text strings, PIL Images, image paths, or video paths)
            modality: Type of content ('text', 'image', 'video')
            **kwargs: Additional arguments for specific encoders
        
        Returns:
            numpy array of embeddings
        """
        if modality == "text":
            return self.encode_text(content, **kwargs)  # type: ignore
        elif modality == "image":
            return self.encode_image(content, **kwargs)  # type: ignore
        elif modality == "video":
            return self.encode_video(content, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported modality: {modality}. Choose from 'text', 'image', 'video'")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        # Test with a simple text input to get embedding dimension
        with torch.no_grad():
            test_input = self.processor(
                text="test",
                images=None,
                return_tensors="pt"
            )
            test_input = {key: value.to(self.device) for key, value in test_input.items()}
            output = self.model(tgt=test_input)["tgt_reps"]
            return output.shape[-1]