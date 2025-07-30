import pandas as pd
import pytest
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.retrieval_debugger import RetrievalDebugger

# Load test data
df = pd.read_csv("data/ragas_outputs/ragas_results.csv")

# Map column names to match the expected format
df = df.rename(columns={
    'user_input': 'question',
    'retrieved_contexts': 'retrieved_chunks', 
    'reference': 'ground_truth'
})

class TestEnhancedRAGASMetrics:
    """Enhanced RAGAS metrics testing with debugging capabilities"""
    
    @pytest.fixture(scope="class")
    def debugger(self):
        """Initialize retrieval debugger"""
        return RetrievalDebugger()
    
    def assert_metric_with_debug(self, metric_series, metric_name: str, threshold: float, 
                                debugger: RetrievalDebugger, debug_failures: bool = True):
        """
        Assert metric with enhanced debugging for failures
        
        Args:
            metric_series: Series of metric values
            metric_name: Name of the metric
            threshold: Minimum acceptable value
            debugger: RetrievalDebugger instance
            debug_failures: Whether to generate debug reports for failures
        """
        failing_indices = []
        
        for idx, value in enumerate(metric_series):
            if value < threshold:
                failing_indices.append(idx)
        
        if failing_indices and debug_failures:
            print(f"\n🔍 DEBUGGING {len(failing_indices)} FAILURES FOR {metric_name.upper()}")
            print("=" * 60)
            
            for idx in failing_indices[:3]:  # Debug first 3 failures
                self._debug_single_failure(idx, metric_name, df, debugger)
        
        # Assert all values meet threshold
        for idx, value in enumerate(metric_series):
            assert value >= threshold, (
                f"{metric_name} too low at index {idx}: {value:.3f} < {threshold} "
                f"→ Q: {df['question'][idx]} | A: {df['response'][idx]}"
            )
    
    def _debug_single_failure(self, idx: int, metric_name: str, df: pd.DataFrame, debugger: RetrievalDebugger):
        """Debug a single failing test case"""
        question = df['question'][idx]
        response = df['response'][idx]
        retrieved_chunks = df['retrieved_chunks'][idx]
        ground_truth = df['ground_truth'][idx]
        
        print(f"\n FAILURE #{idx + 1}: {metric_name}")
        print(f"Question: {question}")
        print(f"Response: {response[:100]}...")
        
        # Analyze retrieval if it's a context-related metric
        if metric_name in ['context_precision', 'context_recall']:
            try:
                # Convert string representation back to list if needed
                if isinstance(retrieved_chunks, str):
                    import ast
                    retrieved_chunks = ast.literal_eval(retrieved_chunks)
                
                # Create ground truth docs list (simplified)
                ground_truth_docs = [ground_truth] if isinstance(ground_truth, str) else ground_truth
                
                # Run retrieval analysis
                analysis = debugger.analyze_retrieval_quality(
                    question=question,
                    ground_truth_docs=ground_truth_docs,
                    k=len(retrieved_chunks) if retrieved_chunks else 4
                )
                
                # Generate debug report
                debug_report = debugger.generate_debug_report(analysis)
                print(debug_report)
                
            except Exception as e:
                print(f"⚠️  Debug analysis failed: {e}")
        
        print("-" * 40)
    
    def test_enhanced_faithfulness_above_0_5(self, debugger):
        """Test faithfulness with enhanced debugging"""
        self.assert_metric_with_debug(
            df["faithfulness"], 
            "faithfulness", 
            0.5, 
            debugger,
            debug_failures=True
        )
    
    def test_enhanced_answer_relevancy_above_0_7(self, debugger):
        """Test answer relevancy with enhanced debugging"""
        self.assert_metric_with_debug(
            df["answer_relevancy"], 
            "answer_relevancy", 
            0.7, 
            debugger,
            debug_failures=True
        )
    
    def test_enhanced_context_precision_above_0_6(self, debugger):
        """Test context precision with enhanced debugging"""
        self.assert_metric_with_debug(
            df["context_precision"], 
            "context_precision", 
            0.6, 
            debugger,
            debug_failures=True
        )
    
    def test_enhanced_context_recall_above_0_6(self, debugger):
        """Test context recall with enhanced debugging"""
        self.assert_metric_with_debug(
            df["context_recall"], 
            "context_recall", 
            0.6, 
            debugger,
            debug_failures=True
        )
    
    def test_retrieval_quality_analysis(self, debugger):
        """Test retrieval quality analysis on sample questions"""
        # Test on a few sample questions
        sample_questions = df['question'].head(3).tolist()
        
        for i, question in enumerate(sample_questions):
            print(f"\n ANALYZING RETRIEVAL FOR QUESTION {i+1}")
            print(f"Question: {question}")
            
            # Get ground truth (simplified)
            ground_truth = df.iloc[i]['ground_truth']
            ground_truth_docs = [ground_truth] if isinstance(ground_truth, str) else ground_truth
            
            # Analyze retrieval
            analysis = debugger.analyze_retrieval_quality(
                question=question,
                ground_truth_docs=ground_truth_docs,
                k=4
            )
            
            # Print key metrics
            print(f"Retrieved: {analysis['retrieved_count']} docs")
            print(f"Ground Truth: {analysis['ground_truth_count']} docs")
            print(f"Recall: {analysis['ground_truth_overlap']['recall']:.3f}")
            print(f"Precision: {analysis['ground_truth_overlap']['precision']:.3f}")
            print(f"Avg Similarity: {sum(analysis['retrieval_scores'])/len(analysis['retrieval_scores']):.3f}")
            
            # Assert reasonable retrieval quality
            assert analysis['ground_truth_overlap']['recall'] > 0.3, f"Low recall for question {i+1}"
            assert analysis['ground_truth_overlap']['precision'] > 0.3, f"Low precision for question {i+1}"
    
    def test_chunk_analysis(self, debugger):
        """Test chunk analysis functionality"""
        question = df['question'].iloc[0]
        ground_truth = df['ground_truth'].iloc[0]
        ground_truth_docs = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        
        analysis = debugger.analyze_retrieval_quality(
            question=question,
            ground_truth_docs=ground_truth_docs,
            k=4
        )
        
        # Test chunk analysis
        chunk_analysis = analysis['chunk_analysis']
        assert chunk_analysis['avg_chunk_size'] > 0, "Chunk size should be positive"
        assert chunk_analysis['total_tokens'] > 0, "Total tokens should be positive"
        assert chunk_analysis['min_chunk_size'] <= chunk_analysis['max_chunk_size'], "Min should be <= max"
        
        print(f"\n📏 CHUNK ANALYSIS RESULTS:")
        print(f"Average chunk size: {chunk_analysis['avg_chunk_size']:.1f} words")
        print(f"Chunk size range: {chunk_analysis['min_chunk_size']} - {chunk_analysis['max_chunk_size']} words")
        print(f"Total tokens: {chunk_analysis['total_tokens']}")
        print(f"Chunk size std: {chunk_analysis['chunk_size_std']:.1f}") 