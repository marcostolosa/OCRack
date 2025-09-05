"""Smart text chunking algorithm for optimal translation."""

import re
from typing import List


def smart_chunks(text: str, max_chars: int = 12000, merge_threshold: int = 1200) -> List[str]:
    """
    Split text into smart chunks suitable for translation.
    
    Algorithm:
    1. Split text into paragraphs
    2. Merge small paragraphs until reaching merge_threshold
    3. Create chunks respecting max_chars limit
    4. Avoid breaking in the middle of sentences, lists, or code blocks
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        merge_threshold: Minimum size to accumulate before closing a chunk
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into paragraphs (double newlines or single newlines with significant content)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed max_chars
        potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        if len(potential_chunk) <= max_chars:
            # Paragraph fits, add it
            current_chunk = potential_chunk
        else:
            # Paragraph doesn't fit
            if current_chunk:
                # If we have accumulated enough text, finalize current chunk
                if len(current_chunk) >= merge_threshold:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    # Current chunk is too small, try to split the paragraph
                    if len(paragraph) > max_chars:
                        # Very long paragraph, need to split it
                        split_parts = split_large_paragraph(paragraph, max_chars - len(current_chunk) - 2)
                        
                        # Add first part to current chunk
                        if current_chunk:
                            chunks.append(current_chunk + "\n\n" + split_parts[0])
                        else:
                            chunks.append(split_parts[0])
                        
                        # Add remaining parts as separate chunks
                        for part in split_parts[1:]:
                            if len(part) > max_chars:
                                # Still too big, split further
                                subparts = split_large_paragraph(part, max_chars)
                                chunks.extend(subparts)
                            else:
                                chunks.append(part)
                        
                        current_chunk = ""
                    else:
                        # Paragraph is reasonable size, finalize current chunk and start new one
                        chunks.append(current_chunk)
                        current_chunk = paragraph
            else:
                # No current chunk, but paragraph is too large
                if len(paragraph) > max_chars:
                    parts = split_large_paragraph(paragraph, max_chars)
                    chunks.extend(parts)
                else:
                    current_chunk = paragraph
    
    # Add any remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-process to merge very small chunks
    return merge_small_chunks(chunks, merge_threshold, max_chars)


def split_large_paragraph(paragraph: str, max_size: int) -> List[str]:
    """
    Split a large paragraph into smaller parts while preserving structure.
    
    Args:
        paragraph: The paragraph to split
        max_size: Maximum size per part
        
    Returns:
        List of paragraph parts
    """
    if len(paragraph) <= max_size:
        return [paragraph]
    
    # Try to split on sentences first
    sentences = split_sentences(paragraph)
    
    if len(sentences) <= 1:
        # No sentence boundaries found, split on word boundaries
        return split_on_words(paragraph, max_size)
    
    parts = []
    current_part = ""
    
    for sentence in sentences:
        potential_part = current_part + " " + sentence if current_part else sentence
        
        if len(potential_part) <= max_size:
            current_part = potential_part
        else:
            if current_part:
                parts.append(current_part.strip())
                
            # Check if sentence itself is too long
            if len(sentence) > max_size:
                # Split the long sentence on words
                word_parts = split_on_words(sentence, max_size)
                parts.extend(word_parts[:-1])  # Add all but last
                current_part = word_parts[-1]   # Last becomes current
            else:
                current_part = sentence
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    # Simple sentence splitting - could be improved with nltk or spacy
    sentence_endings = re.compile(r'([.!?]+)(\s+)')
    
    sentences = []
    start = 0
    
    for match in sentence_endings.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    
    # Add remaining text
    remaining = text[start:].strip()
    if remaining:
        sentences.append(remaining)
    
    return sentences


def split_on_words(text: str, max_size: int) -> List[str]:
    """Split text on word boundaries."""
    words = text.split()
    parts = []
    current_part = ""
    
    for word in words:
        potential_part = current_part + " " + word if current_part else word
        
        if len(potential_part) <= max_size:
            current_part = potential_part
        else:
            if current_part:
                parts.append(current_part)
            
            # If single word is too long, truncate it (rare case)
            if len(word) > max_size:
                parts.append(word[:max_size])
                current_part = word[max_size:]
            else:
                current_part = word
    
    if current_part:
        parts.append(current_part)
    
    return parts


def merge_small_chunks(chunks: List[str], min_size: int, max_size: int) -> List[str]:
    """
    Merge consecutive small chunks to avoid too many tiny API calls.
    
    Args:
        chunks: List of text chunks
        min_size: Minimum preferred chunk size
        max_size: Maximum allowed chunk size
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks
    
    merged = []
    current_merged = ""
    
    for chunk in chunks:
        # Try to merge with current
        potential_merged = current_merged + "\n\n" + chunk if current_merged else chunk
        
        if len(potential_merged) <= max_size:
            # Can merge
            current_merged = potential_merged
        else:
            # Cannot merge, finalize current
            if current_merged:
                merged.append(current_merged)
            current_merged = chunk
    
    # Add final chunk
    if current_merged:
        merged.append(current_merged)
    
    return merged


def count_chunks(text: str, max_chars: int = 12000, merge_threshold: int = 1200) -> int:
    """
    Count how many chunks the text would be split into.
    
    Args:
        text: Input text
        max_chars: Maximum characters per chunk
        merge_threshold: Minimum size before finalizing chunk
        
    Returns:
        Number of chunks
    """
    chunks = smart_chunks(text, max_chars, merge_threshold)
    return len(chunks)


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of tokens (for cost calculation).
    Uses approximation of ~4 characters per token.
    
    Args:
        text: Input text
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // 4


def chunk_statistics(chunks: List[str]) -> dict:
    """
    Generate statistics about chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'count': 0,
            'total_chars': 0,
            'avg_chars': 0,
            'min_chars': 0,
            'max_chars': 0,
            'estimated_tokens': 0
        }
    
    char_counts = [len(chunk) for chunk in chunks]
    
    return {
        'count': len(chunks),
        'total_chars': sum(char_counts),
        'avg_chars': sum(char_counts) // len(char_counts),
        'min_chars': min(char_counts),
        'max_chars': max(char_counts),
        'estimated_tokens': sum(estimate_tokens(chunk) for chunk in chunks)
    }