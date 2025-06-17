
"""
Temporal Query Classifier for CRISPR Historian
"""

from typing import Dict, Any
from langchain.prompts import PromptTemplate


class TemporalQueryClassifier:
    """Classifies queries into temporal categories"""
    
    def __init__(self, llm):
        self.llm = llm
        self.classification_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Classify this query into one of these three categories:

EXPLICIT_TEMPORAL: Contains specific years, dates, or time periods (e.g., "in 1995", "by 2000", "during the 1990s", "between 1985-1990")

EVOLUTIONARY: Asks about development, evolution, history, evidence, discovery process, or "how we know" something. These questions seek the journey of scientific understanding over time.

STANDARD: Current factual question without temporal aspect - wants present knowledge.

Examples:
- "What hypotheses did Mojica propose in 1995?" → EXPLICIT_TEMPORAL
- "How do we know CRISPR is an adaptive immune system?" → EVOLUTIONARY  
- "Timeline of CRISPR discoveries" → EVOLUTIONARY
- "What evidence supports CRISPR function?" → EVOLUTIONARY
- "What is Cas9?" → STANDARD
- "How does CRISPR work?" → STANDARD

Question: {question}

Classification (respond with only one word - EXPLICIT_TEMPORAL, EVOLUTIONARY, or STANDARD):"""
        )
    
    def classify_query(self, question: str) -> Dict[str, Any]:
        """Classify a query and return classification with reasoning"""
        try:
            # Get classification
            response = self.llm.invoke(self.classification_prompt.format(question=question))
            classification = response.content.strip().upper()
            
            # Validate classification
            valid_classifications = ["EXPLICIT_TEMPORAL", "EVOLUTIONARY", "STANDARD"]
            if classification not in valid_classifications:
                # Try to extract valid classification from response
                for valid_class in valid_classifications:
                    if valid_class in classification:
                        classification = valid_class
                        break
                else:
                    classification = "STANDARD"  # Default fallback
            
            # Generate reasoning (separate call for explanation)
            reasoning_prompt = f"""Explain why this question was classified as {classification}:

Question: {question}
Classification: {classification}

Provide a brief explanation (1-2 sentences) of why this classification was chosen:"""
            
            reasoning_response = self.llm.invoke(reasoning_prompt)
            reasoning = reasoning_response.content.strip()
            
            return {
                "classification": classification,
                "reasoning": reasoning,
                "confidence": "high" if classification in valid_classifications else "low"
            }
            
        except Exception as e:  # pylint: disable=broad-except
            return {
                "classification": "STANDARD",
                "reasoning": f"Error in classification: {str(e)}",
                "confidence": "low"
            }
