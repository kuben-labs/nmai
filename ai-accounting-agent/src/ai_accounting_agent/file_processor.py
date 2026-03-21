"""File processor for extracting data from PDFs, images, and other attachments."""

import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


class FileProcessor:
    """Process and extract data from attached files (PDFs, images, etc)."""

    @staticmethod
    def decode_base64_file(content_base64: str) -> bytes:
        """
        Decode a base64-encoded file.

        Args:
            content_base64: Base64-encoded file content

        Returns:
            Decoded file bytes
        """
        try:
            return base64.b64decode(content_base64)
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            raise ValueError(f"Invalid base64 content: {e}") from e

    @staticmethod
    def save_file(filename: str, file_bytes: bytes, temp_dir: str = "/tmp") -> str:
        """
        Save a file to disk.

        Args:
            filename: Original filename
            file_bytes: File content as bytes
            temp_dir: Directory to save file to

        Returns:
            Path to saved file
        """
        try:
            # Create temp directory if it doesn't exist
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            # Save file - create parent dirs in case filename includes subdirectories
            # (e.g., "files/tilbudsbrev_nn_08.pdf")
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_bytes)

            logger.debug(f"Saved file: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise

    @staticmethod
    def process_attachment(
        filename: str, content_base64: str, mime_type: str
    ) -> Dict[str, Any]:
        """
        Process a single attached file.

        Args:
            filename: Original filename
            content_base64: Base64-encoded content
            mime_type: MIME type of the file

        Returns:
            Dictionary with file metadata and extracted data
        """
        try:
            logger.info(f"Processing file: {filename} ({mime_type})")

            # Decode file
            file_bytes = FileProcessor.decode_base64_file(content_base64)
            file_size = len(file_bytes)

            logger.debug(f"File size: {file_size} bytes")

            # Save to temp location
            file_path = FileProcessor.save_file(filename, file_bytes)

            # Extract text/data based on mime type
            extracted_data = FileProcessor._extract_content(
                file_path, filename, mime_type, file_bytes
            )

            return {
                "success": True,
                "filename": filename,
                "mime_type": mime_type,
                "size": file_size,
                "path": file_path,
                "extracted_data": extracted_data,
            }

        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {e}")
            return {"success": False, "filename": filename, "error": str(e)}

    @staticmethod
    def _extract_content(
        file_path: str, filename: str, mime_type: str, file_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Extract content from a file based on its type.

        Args:
            file_path: Path to the file
            filename: Original filename
            mime_type: MIME type
            file_bytes: Raw file bytes

        Returns:
            Dictionary with extracted content
        """
        try:
            if mime_type == "application/pdf":
                return FileProcessor._extract_pdf(file_path)

            elif mime_type.startswith("image/"):
                return FileProcessor._extract_image(file_path, mime_type)

            elif mime_type in ["text/plain", "text/csv"]:
                return {
                    "type": "text",
                    "content": file_bytes.decode("utf-8", errors="ignore"),
                }

            else:
                # Unknown format
                logger.warning(f"Unknown mime type: {mime_type}")
                return {
                    "type": "unknown",
                    "message": f"File type {mime_type} not yet supported for extraction",
                }

        except Exception as e:
            logger.error(f"Error extracting content from {filename}: {e}")
            return {"type": "error", "error": str(e)}

    @staticmethod
    def _extract_pdf(file_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with extracted PDF data
        """
        try:
            # Try importing pdfplumber
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    pages_data = []
                    full_text = []

                    for i, page in enumerate(pdf.pages):
                        # Extract text
                        text = page.extract_text()
                        full_text.append(text)

                        # Try to extract tables
                        tables = page.extract_tables()

                        pages_data.append(
                            {
                                "page_number": i + 1,
                                "text": text,
                                "has_tables": len(tables) > 0 if tables else False,
                                "table_count": len(tables) if tables else 0,
                            }
                        )

                    return {
                        "type": "pdf",
                        "page_count": len(pdf.pages),
                        "full_text": "\n\n---PAGE BREAK---\n\n".join(full_text),
                        "pages": pages_data,
                        "message": "PDF text extracted. Use full_text for LLM processing.",
                    }

            except ImportError:
                logger.warning("pdfplumber not installed, attempting fallback...")

                # Fallback: Try PyPDF2
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(file_path)
                    pages_data = []
                    full_text = []

                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        full_text.append(text)
                        pages_data.append({"page_number": i + 1, "text": text})

                    return {
                        "type": "pdf",
                        "page_count": len(reader.pages),
                        "full_text": "\n\n---PAGE BREAK---\n\n".join(full_text),
                        "pages": pages_data,
                        "message": "PDF text extracted (PyPDF2). Some formatting may be lost.",
                    }

                except ImportError:
                    logger.warning("No PDF library available")
                    return {
                        "type": "pdf",
                        "error": "PDF extraction requires pdfplumber or PyPDF2",
                        "message": "Please install: pip install pdfplumber",
                    }

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return {"type": "pdf", "error": str(e)}

    @staticmethod
    def _extract_image(file_path: str, mime_type: str) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.

        Args:
            file_path: Path to the image file
            mime_type: MIME type of the image

        Returns:
            Dictionary with extracted image data
        """
        try:
            # Try importing pytesseract
            try:
                import pytesseract
                from PIL import Image

                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)

                return {
                    "type": "image",
                    "mime_type": mime_type,
                    "text": text,
                    "size": f"{image.width}x{image.height}",
                    "message": "OCR text extracted. Accuracy depends on image quality.",
                }

            except ImportError:
                logger.warning("pytesseract not installed")
                return {
                    "type": "image",
                    "error": "OCR requires pytesseract",
                    "message": "Please install: pip install pytesseract pillow",
                    "instructions": "Also requires Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki",
                }

        except Exception as e:
            logger.error(f"Error extracting image: {e}")
            return {"type": "image", "error": str(e)}

    @staticmethod
    def process_files(files: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process all attached files.

        Args:
            files: List of file dictionaries with filename, content_base64, mime_type

        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {len(files)} files...")

        results = {
            "success": True,
            "total_files": len(files),
            "processed_files": [],
            "failed_files": [],
            "extracted_data": {},
        }

        for file_data in files:
            try:
                filename = file_data.get("filename")
                content_base64 = file_data.get("content_base64")
                mime_type = file_data.get("mime_type", "application/octet-stream")

                if not filename or not content_base64:
                    logger.warning("File missing filename or content")
                    results["failed_files"].append(
                        {
                            "filename": filename or "unknown",
                            "error": "Missing filename or content",
                        }
                    )
                    continue

                # Process file
                result = FileProcessor.process_attachment(
                    filename, content_base64, mime_type
                )

                if result.get("success"):
                    results["processed_files"].append(result)
                    results["extracted_data"][filename] = result.get(
                        "extracted_data", {}
                    )
                else:
                    results["failed_files"].append(result)
                    results["success"] = False

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                results["failed_files"].append(
                    {"filename": file_data.get("filename", "unknown"), "error": str(e)}
                )
                results["success"] = False

        logger.info(
            f"File processing complete: "
            f"{len(results['processed_files'])} successful, "
            f"{len(results['failed_files'])} failed"
        )

        return results
