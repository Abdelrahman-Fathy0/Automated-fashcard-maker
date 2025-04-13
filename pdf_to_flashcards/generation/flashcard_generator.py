#!/usr/bin/env python3
"""
Flashcard generator base class.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Flashcard:
    """
    Representation of a flashcard with question and answer.
    """
    question: str
    answer: str
    source_chunk_id: Optional[int] = None
    source_pages: List[int] = field(default_factory=list)
    quality_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flashcard to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "source_chunk_id": self.source_chunk_id,
            "source_pages": self.source_pages,
            "quality_score": self.quality_score,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Flashcard":
        """Create flashcard from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            source_chunk_id=data.get("source_chunk_id"),
            source_pages=data.get("source_pages", []),
            quality_score=data.get("quality_score", 0.0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
