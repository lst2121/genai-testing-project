import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import config
import numpy as np

class RetrievalDebugger:
    """Debug retrieval issues in RAG pipeline"""
    
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=config.PERSIST_DIRECTORY,
            collection_name=config.EMBEDDING_COLLECTION_NAME
        )
        self.retriever = self.vectorstore.as_retriever()
    
    def analyze_retrieval_quality(self, question: str, ground_truth_docs: List[str], 
                                 k: int = 4) -> Dict[str, Any]:
        """
        Analyze retrieval quality for a specific question
        
        Args:
            question: User question
            ground_truth_docs: List of expected document content
            k: Number of docs to retrieve
            
        Returns:
            Dict with retrieval analysis
        """
        # Get retrieved documents
        retrieved_docs = self.retriever.get_relevant_documents(question, k=k)
        
        # Calculate retrieval scores
        retrieval_scores = self._get_retrieval_scores(question, retrieved_docs)
        
        # Analyze chunk characteristics
        chunk_analysis = self._analyze_chunks(retrieved_docs)
        
        # Calculate overlap with ground truth
        ground_truth_overlap = self._calculate_ground_truth_overlap(
            retrieved_docs, ground_truth_docs
        )
        
        return {
            "question": question,
            "retrieved_count": len(retrieved_docs),
            "ground_truth_count": len(ground_truth_docs),
            "retrieval_scores": retrieval_scores,
            "chunk_analysis": chunk_analysis,
            "ground_truth_overlap": ground_truth_overlap,
            "retrieved_docs": [doc.page_content for doc in retrieved_docs],
            "missing_docs": self._find_missing_docs(retrieved_docs, ground_truth_docs)
        }
    
    def _get_retrieval_scores(self, question: str, docs: List[Document]) -> List[float]:
        """Get similarity scores for retrieved documents"""
        question_embedding = self.embedding_model.embed_query(question)
        scores = []
        
        for doc in docs:
            doc_embedding = self.embedding_model.embed_query(doc.page_content)
            # Calculate cosine similarity
            similarity = np.dot(question_embedding, doc_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append(similarity)
        
        return scores
    
    def _analyze_chunks(self, docs: List[Document]) -> Dict[str, Any]:
        """Analyze chunk characteristics"""
        chunk_sizes = [len(doc.page_content.split()) for doc in docs]
        
        return {
            "avg_chunk_size": np.mean(chunk_sizes),
            "min_chunk_size": np.min(chunk_sizes),
            "max_chunk_size": np.max(chunk_sizes),
            "chunk_size_std": np.std(chunk_sizes),
            "total_tokens": sum(chunk_sizes)
        }
    
    def _calculate_ground_truth_overlap(self, retrieved_docs: List[Document], 
                                      ground_truth_docs: List[str]) -> Dict[str, float]:
        """Calculate overlap between retrieved and ground truth documents"""
        retrieved_content = [doc.page_content for doc in retrieved_docs]
        
        # Simple content-based overlap (you could make this more sophisticated)
        overlap_count = 0
        for gt_doc in ground_truth_docs:
            for ret_doc in retrieved_content:
                if any(keyword in ret_doc.lower() for keyword in gt_doc.lower().split()[:5]):
                    overlap_count += 1
                    break
        
        precision = overlap_count / len(retrieved_docs) if retrieved_docs else 0
        recall = overlap_count / len(ground_truth_docs) if ground_truth_docs else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "overlap_count": overlap_count
        }
    
    def _find_missing_docs(self, retrieved_docs: List[Document], 
                          ground_truth_docs: List[str]) -> List[str]:
        """Find ground truth documents that weren't retrieved"""
        retrieved_content = [doc.page_content for doc in retrieved_docs]
        missing = []
        
        for gt_doc in ground_truth_docs:
            found = False
            for ret_doc in retrieved_content:
                if any(keyword in ret_doc.lower() for keyword in gt_doc.lower().split()[:5]):
                    found = True
                    break
            if not found:
                missing.append(gt_doc)
        
        return missing
    
    def generate_debug_report(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable debug report"""
        report = f"""
RETRIEVAL DEBUG REPORT
========================

Question: {analysis['question']}

RETRIEVAL METRICS:
- Retrieved: {analysis['retrieved_count']} docs
- Ground Truth: {analysis['ground_truth_count']} docs
- Precision: {analysis['ground_truth_overlap']['precision']:.3f}
- Recall: {analysis['ground_truth_overlap']['recall']:.3f}

CHUNK ANALYSIS:
- Avg Chunk Size: {analysis['chunk_analysis']['avg_chunk_size']:.1f} words
- Chunk Size Range: {analysis['chunk_analysis']['min_chunk_size']} - {analysis['chunk_analysis']['max_chunk_size']} words
- Total Tokens: {analysis['chunk_analysis']['total_tokens']}

SIMILARITY SCORES:
- Min: {min(analysis['retrieval_scores']):.3f}
- Max: {max(analysis['retrieval_scores']):.3f}
- Avg: {np.mean(analysis['retrieval_scores']):.3f}

MISSING DOCUMENTS ({len(analysis['missing_docs'])}):
"""
        
        for i, missing_doc in enumerate(analysis['missing_docs'], 1):
            report += f"{i}. {missing_doc[:100]}...\n"
        
        # Add recommendations
        report += "\n💡 RECOMMENDATIONS:\n"
        if analysis['ground_truth_overlap']['recall'] < 0.5:
            report += "- Low recall: Consider increasing chunk overlap or adjusting retrieval threshold\n"
        if analysis['chunk_analysis']['avg_chunk_size'] < 50:
            report += "- Small chunks: Consider increasing chunk size\n"
        if max(analysis['retrieval_scores']) < 0.7:
            report += "- Low similarity scores: Check embedding model or query formulation\n"
        
        return report 