from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import numpy as np
from PIL import Image


@dataclass
class MbodiClient:
    """
    HTTPX client for the Mbodi AI FastAPI server.

    Designed so that:
    - infer(...) accepts the same logical args as the inference endpoint
    - act(...) accepts the same logical args as the data collector endpoint
    """

    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MbodiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3:
            raise ValueError(f"image must have shape (H, W, C), got {image.shape}")
        if image.shape[2] != 3:
            raise ValueError(f"image must have 3 channels, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"image dtype must be np.uint8, got {image.dtype}")

    @staticmethod
    def _image_to_bytes(
        image: np.ndarray,
        image_format: str = "PNG",
    ) -> tuple[str, bytes, str]:
        """
        Convert np.ndarray[(H, W, 3), uint8] to uploadable image bytes.
        Returns: (filename, content, mime_type)
        """
        MbodiClient._validate_image(image)

        pil_img = Image.fromarray(image)

        buffer = io.BytesIO()
        fmt = image_format.upper()

        if fmt == "PNG":
            pil_img.save(buffer, format="PNG")
            return ("image.png", buffer.getvalue(), "image/png")
        elif fmt in {"JPG", "JPEG"}:
            pil_img.save(buffer, format="JPEG", quality=95)
            return ("image.jpg", buffer.getvalue(), "image/jpeg")
        else:
            raise ValueError("image_format must be 'PNG' or 'JPEG'")

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def infer(
        self,
        image: np.ndarray,
        labels: list[str],
        model_file_path: Optional[str] = None,
        image_format: str = "PNG",
    ) -> dict[str, Any]:
        """
        Calls POST /model_inference

        Parameters match the intended inference pipeline usage:
        - image: np.ndarray[(H, W, C), np.uint8]
        - labels: list[str]
        - model_file_path: optional override
        """
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise TypeError("labels must be a list[str]")

        filename, image_bytes, mime_type = self._image_to_bytes(image, image_format=image_format)

        files = {
            "image": (filename, image_bytes, mime_type),
        }

        data = {
            # your server currently does labels_json.split(","),
            # so comma-joined strings are what it actually expects
            "labels_json": ",".join(labels),
        }

        if model_file_path is not None:
            data["model_file_path"] = model_file_path

        response = self._client.post(
            "/model_inference",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.json()

    def act(
        self,
        image: np.ndarray,
        bounding_boxes: dict[str, list[list[tuple[int, int]]]],
        message: str,
        coco_root: Optional[str] = None,
        forced_inclusions_dir: Optional[str] = None,
        base_dataset_dir: Optional[str] = None,
        images_zip_name: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        image_format: str = "PNG",
    ) -> dict[str, Any]:
        """
        Calls POST /data_collector

        Parameters match the intended data collector usage:
        - image: np.ndarray[(H, W, C), np.uint8]
        - bounding_boxes: dict[str, list[list[tuple[int, int]]]]
        - message: str
        """
        if not isinstance(message, str):
            raise TypeError("message must be a str")
        if not isinstance(bounding_boxes, dict):
            raise TypeError("bounding_boxes must be a dict")

        filename, image_bytes, mime_type = self._image_to_bytes(image, image_format=image_format)

        files = {
            "image": (filename, image_bytes, mime_type),
        }

        data: dict[str, Any] = {
            "bounding_boxes_json": json.dumps(bounding_boxes),
            "message": message,
        }

        if coco_root is not None:
            data["coco_root"] = coco_root
        if forced_inclusions_dir is not None:
            data["forced_inclusions_dir"] = forced_inclusions_dir
        if base_dataset_dir is not None:
            data["base_dataset_dir"] = base_dataset_dir
        if images_zip_name is not None:
            data["images_zip_name"] = images_zip_name
        if jpeg_quality is not None:
            data["jpeg_quality"] = str(jpeg_quality)

        response = self._client.post(
            "/data_collector",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.json()