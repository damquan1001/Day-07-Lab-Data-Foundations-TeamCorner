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
        
        # Split on sentence boundaries using regex lookbehind
        # Delimiters: ". ", "! ", "? ", ".\n"
        pattern = r'(?<=\. )|(?<=! )|(?<=\? )|(?<=\.\n)'
        raw_sentences = re.split(pattern, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        if not sentences:
            return []
            
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
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
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fall back gracefully: split into character chunks
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Find the first separator that actually exists in current_text
        separator = None
        next_idx = 0
        for idx, sep in enumerate(remaining_separators):
            if sep == "":
                separator = ""
                next_idx = idx + 1
                break
            if sep in current_text:
                separator = sep
                next_idx = idx + 1
                break

        if separator is None:
            # Fall back to character chunks if no separator is found
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Split current_text by the chosen separator
        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        # Recursively split any pieces that are too large
        next_seps = remaining_separators[next_idx:]
        final_splits = []
        for s in splits:
            if len(s) <= self.chunk_size:
                final_splits.append(s)
            else:
                final_splits.extend(self._split(s, next_seps))

        # Merge the final_splits using the separator, without exceeding chunk_size
        merged_chunks = []
        current_chunk = []
        current_len = 0

        for split in final_splits:
            if not split:
                continue
            sep_len = len(separator) if current_chunk and separator != "" else 0
            if current_len + sep_len + len(split) <= self.chunk_size:
                current_chunk.append(split)
                current_len += sep_len + len(split)
            else:
                if current_chunk:
                    merged_chunks.append(separator.join(current_chunk))
                current_chunk = [split]
                current_len = len(split)

        if current_chunk:
            merged_chunks.append(separator.join(current_chunk))

        return merged_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        results = {}
        for name, chunker in [
            ('fixed_size', fixed_chunker),
            ('by_sentences', sentence_chunker),
            ('recursive', recursive_chunker)
        ]:
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            results[name] = {
                'count': count,
                'avg_length': avg_len,
                'chunks': chunks
            }
        return results
