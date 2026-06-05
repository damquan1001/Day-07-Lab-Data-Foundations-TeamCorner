from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+|\n+', text) if sentence.strip()]
        chunks: list[str] = []
        current_chunk: list[str] = []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        raw_chunks = self._split(text, self.separators)
        return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text]

        separator = remaining_separators[0]
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        pieces = current_text.split(separator)
        if len(pieces) == 1:
            return self._split(current_text, remaining_separators[1:])

        normalized_pieces: list[str] = []
        for index, piece in enumerate(pieces):
            if index < len(pieces) - 1:
                normalized_pieces.append(piece + separator)
            else:
                normalized_pieces.append(piece)

        oversized = [piece for piece in normalized_pieces if len(piece) > self.chunk_size]
        if oversized and len(remaining_separators) > 1:
            split_chunks: list[str] = []
            for piece in normalized_pieces:
                if len(piece) > self.chunk_size:
                    split_chunks.extend(self._split(piece, remaining_separators[1:]))
                elif piece:
                    split_chunks.append(piece)
            return split_chunks

        if all(len(piece) <= self.chunk_size for piece in normalized_pieces):
            return normalized_pieces

        return self._split(current_text, remaining_separators[1:])


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
#     """
#     Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(sum(value * value for value in vec_a))
    magnitude_b = math.sqrt(sum(value * value for value in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)

#     Returns 0.0 if either vector has zero magnitude.
#     """
#     # TODO: implement cosine similarity formula
#     raise NotImplementedError("Implement compute_similarity")
def compute_similarity(vec_a, vec_b):
    mag_a = math.sqrt(sum(x*x for x in vec_a))
    mag_b = math.sqrt(sum(x*x for x in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)

class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_size_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, min(50, chunk_size // 10))).chunk(text)
        by_sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
            return {
                "chunks": chunks,
                "count": count,
                "avg_length": avg_length,
            }

        return {
            "fixed_size": _stats(fixed_size_chunks),
            "by_sentences": _stats(by_sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
