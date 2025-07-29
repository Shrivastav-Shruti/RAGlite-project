"""
Performance benchmarking module for RAGLite.

Provides comprehensive performance testing and analysis capabilities.
"""

import time
import logging
import statistics
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class RAGBenchmark:
    """
    Comprehensive benchmarking suite for RAGLite performance analysis.
    """
    
    def __init__(self, 
                 retriever=None, 
                 groq_client=None,
                 output_dir: str = "./benchmarks"):
        """
        Initialize the benchmark suite.
        
        Args:
            retriever: RAGLite Retriever instance
            groq_client: GroqClient instance
            output_dir: Directory to save benchmark results
        """
        self.retriever = retriever
        self.groq_client = groq_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def benchmark_document_loading(self, 
                                  document_paths: List[str],
                                  chunk_sizes: List[int] = [250, 500, 1000]) -> Dict[str, Any]:
        """
        Benchmark document loading performance with different chunk sizes.
        
        Args:
            document_paths: List of document file paths to test
            chunk_sizes: List of chunk sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        from ..loaders import DocumentLoader
        
        results = {
            'test_name': 'document_loading',
            'timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        for chunk_size in chunk_sizes:
            logger.info(f"Benchmarking document loading with chunk size: {chunk_size}")
            
            chunk_results = {
                'chunk_size': chunk_size,
                'documents': []
            }
            
            loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=50)
            
            for doc_path in document_paths:
                try:
                    start_time = time.time()
                    documents = loader.load_document(doc_path)
                    loading_time = time.time() - start_time
                    
                    doc_result = {
                        'file_path': doc_path,
                        'loading_time': loading_time,
                        'num_chunks': len(documents),
                        'avg_chunk_size': statistics.mean(len(doc['text']) for doc in documents),
                        'total_characters': sum(len(doc['text']) for doc in documents)
                    }
                    
                    chunk_results['documents'].append(doc_result)
                    logger.info(f"Loaded {doc_path}: {len(documents)} chunks in {loading_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error loading {doc_path}: {e}")
                    chunk_results['documents'].append({
                        'file_path': doc_path,
                        'error': str(e)
                    })
            
            results['results'].append(chunk_results)
        
        self.results.append(results)
        return results
    
    def benchmark_embedding_generation(self, 
                                     texts: List[str],
                                     batch_sizes: List[int] = [1, 8, 16, 32]) -> Dict[str, Any]:
        """
        Benchmark embedding generation performance.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.retriever or not self.retriever.embedding_model:
            raise ValueError("Retriever with embedding model required for this benchmark")
        
        results = {
            'test_name': 'embedding_generation',
            'timestamp': datetime.now().isoformat(),
            'num_texts': len(texts),
            'results': []
        }
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking embedding generation with batch size: {batch_size}")
            
            start_time = time.time()
            
            try:
                embeddings = self.retriever.embedding_model.embed_texts(
                    texts, 
                    batch_size=batch_size, 
                    show_progress=False
                )
                
                generation_time = time.time() - start_time
                
                batch_result = {
                    'batch_size': batch_size,
                    'generation_time': generation_time,
                    'texts_per_second': len(texts) / generation_time,
                    'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
                    'success': True
                }
                
                logger.info(f"Generated {len(texts)} embeddings in {generation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error with batch size {batch_size}: {e}")
                batch_result = {
                    'batch_size': batch_size,
                    'error': str(e),
                    'success': False
                }
            
            results['results'].append(batch_result)
        
        self.results.append(results)
        return results
    
    def benchmark_retrieval_performance(self, 
                                      queries: List[str],
                                      top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Benchmark document retrieval performance.
        
        Args:
            queries: List of query strings to test
            top_k_values: List of top-k values to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.retriever:
            raise ValueError("Retriever required for this benchmark")
        
        results = {
            'test_name': 'retrieval_performance',
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'results': []
        }
        
        for top_k in top_k_values:
            logger.info(f"Benchmarking retrieval with top_k: {top_k}")
            
            top_k_results = {
                'top_k': top_k,
                'queries': []
            }
            
            retrieval_times = []
            
            for query in queries:
                try:
                    start_time = time.time()
                    retrieved_docs = self.retriever.retrieve_similar(query, top_k=top_k, similarity_threshold=0.3)
                    retrieval_time = time.time() - start_time
                    
                    retrieval_times.append(retrieval_time)
                    
                    query_result = {
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'retrieval_time': retrieval_time,
                        'num_retrieved': len(retrieved_docs),
                        'avg_similarity': statistics.mean(doc.get('similarity', 0) for doc in retrieved_docs) if retrieved_docs else 0
                    }
                    
                    top_k_results['queries'].append(query_result)
                    
                except Exception as e:
                    logger.error(f"Error retrieving for query '{query[:30]}...': {e}")
                    top_k_results['queries'].append({
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'error': str(e)
                    })
            
            # Calculate summary statistics
            if retrieval_times:
                top_k_results['summary'] = {
                    'avg_retrieval_time': statistics.mean(retrieval_times),
                    'median_retrieval_time': statistics.median(retrieval_times),
                    'min_retrieval_time': min(retrieval_times),
                    'max_retrieval_time': max(retrieval_times),
                    'std_retrieval_time': statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0
                }
            
            results['results'].append(top_k_results)
        
        self.results.append(results)
        return results
    
    def benchmark_end_to_end_rag(self, 
                                query_answer_pairs: List[Tuple[str, str]],
                                temperatures: List[float] = [0.0, 0.2, 0.5]) -> Dict[str, Any]:
        """
        Benchmark end-to-end RAG performance.
        
        Args:
            query_answer_pairs: List of (query, expected_answer) tuples
            temperatures: List of temperature values to test
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.retriever or not self.groq_client:
            raise ValueError("Both retriever and groq_client required for this benchmark")
        
        results = {
            'test_name': 'end_to_end_rag',
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(query_answer_pairs),
            'results': []
        }
        
        for temperature in temperatures:
            logger.info(f"Benchmarking end-to-end RAG with temperature: {temperature}")
            
            temp_results = {
                'temperature': temperature,
                'queries': []
            }
            
            total_times = []
            retrieval_times = []
            generation_times = []
            
            for query, expected_answer in query_answer_pairs:
                try:
                    # Measure retrieval time
                    start_time = time.time()
                    retrieved_docs = self.retriever.retrieve_similar(query, top_k=5, similarity_threshold=0.3)
                    retrieval_time = time.time() - start_time
                    retrieval_times.append(retrieval_time)
                    
                    # Measure generation time
                    start_time = time.time()
                    self.groq_client.temperature = temperature
                    response = self.groq_client.generate_rag_response(query, retrieved_docs)
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                    total_time = retrieval_time + generation_time
                    total_times.append(total_time)
                    
                    query_result = {
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': total_time,
                        'num_retrieved_docs': len(retrieved_docs),
                        'response_length': len(response.get('response', '')),
                        'expected_answer': expected_answer[:100] + "..." if len(expected_answer) > 100 else expected_answer,
                        'actual_response': response.get('response', '')[:100] + "..." if len(response.get('response', '')) > 100 else response.get('response', '')
                    }
                    
                    temp_results['queries'].append(query_result)
                    
                except Exception as e:
                    logger.error(f"Error in end-to-end RAG for query '{query[:30]}...': {e}")
                    temp_results['queries'].append({
                        'query': query[:50] + "..." if len(query) > 50 else query,
                        'error': str(e)
                    })
            
            # Calculate summary statistics
            if total_times:
                temp_results['summary'] = {
                    'avg_total_time': statistics.mean(total_times),
                    'avg_retrieval_time': statistics.mean(retrieval_times),
                    'avg_generation_time': statistics.mean(generation_times),
                    'median_total_time': statistics.median(total_times),
                    'min_total_time': min(total_times),
                    'max_total_time': max(total_times)
                }
            
            results['results'].append(temp_results)
        
        self.results.append(results)
        return results
    
    def benchmark_memory_usage(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark memory usage during different operations.
        
        Args:
            sample_texts: Sample texts for testing
            
        Returns:
            Dictionary with memory usage results
        """
        try:
            import psutil
            import os
        except ImportError:
            logger.warning("psutil not available, skipping memory benchmark")
            return {'error': 'psutil not available'}
        
        process = psutil.Process(os.getpid())
        
        results = {
            'test_name': 'memory_usage',
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        results['results']['baseline_memory_mb'] = baseline_memory
        
        if self.retriever and self.retriever.embedding_model:
            # Memory after loading embedding model
            model_memory = process.memory_info().rss / 1024 / 1024
            results['results']['with_embedding_model_mb'] = model_memory
            results['results']['embedding_model_overhead_mb'] = model_memory - baseline_memory
            
            # Memory during embedding generation
            if sample_texts:
                embeddings = self.retriever.embedding_model.embed_texts(sample_texts[:10])
                embedding_memory = process.memory_info().rss / 1024 / 1024
                results['results']['during_embedding_generation_mb'] = embedding_memory
                results['results']['embedding_generation_overhead_mb'] = embedding_memory - model_memory
        
        self.results.append(results)
        return results
    
    def run_comprehensive_benchmark(self, 
                                  document_paths: List[str],
                                  test_queries: List[str],
                                  query_answer_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark suite.
        
        Args:
            document_paths: Paths to test documents
            test_queries: Test queries for retrieval benchmarking
            query_answer_pairs: Optional query-answer pairs for end-to-end testing
            
        Returns:
            Complete benchmark results
        """
        logger.info("Starting comprehensive benchmark suite")
        
        comprehensive_results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'benchmarks': []
        }
        
        # Document loading benchmark
        if document_paths:
            logger.info("Running document loading benchmark")
            doc_results = self.benchmark_document_loading(document_paths)
            comprehensive_results['benchmarks'].append(doc_results)
        
        # Embedding generation benchmark
        if test_queries:
            logger.info("Running embedding generation benchmark")
            embedding_results = self.benchmark_embedding_generation(test_queries)
            comprehensive_results['benchmarks'].append(embedding_results)
        
        # Retrieval performance benchmark
        if test_queries and self.retriever:
            logger.info("Running retrieval performance benchmark")
            retrieval_results = self.benchmark_retrieval_performance(test_queries)
            comprehensive_results['benchmarks'].append(retrieval_results)
        
        # End-to-end RAG benchmark
        if query_answer_pairs and self.retriever and self.groq_client:
            logger.info("Running end-to-end RAG benchmark")
            e2e_results = self.benchmark_end_to_end_rag(query_answer_pairs)
            comprehensive_results['benchmarks'].append(e2e_results)
        
        # Memory usage benchmark
        if test_queries:
            logger.info("Running memory usage benchmark")
            memory_results = self.benchmark_memory_usage(test_queries)
            comprehensive_results['benchmarks'].append(memory_results)
        
        comprehensive_results['end_time'] = datetime.now().isoformat()
        
        # Save results
        self.save_results(comprehensive_results, f"comprehensive_benchmark_{self.session_id}")
        
        logger.info("Comprehensive benchmark completed")
        return comprehensive_results
    
    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save benchmark results to file.
        
        Args:
            results: Results dictionary to save
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable benchmark report.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "RAGLite Performance Benchmark Report",
            "=" * 60,
            f"Session ID: {results.get('session_id', 'Unknown')}",
            f"Start Time: {results.get('start_time', 'Unknown')}",
            f"End Time: {results.get('end_time', 'Unknown')}",
            ""
        ]
        
        for benchmark in results.get('benchmarks', []):
            test_name = benchmark.get('test_name', 'Unknown Test')
            report_lines.extend([
                f"📊 {test_name.upper().replace('_', ' ')}",
                "-" * 40
            ])
            
            if test_name == 'retrieval_performance':
                for result in benchmark.get('results', []):
                    top_k = result.get('top_k', 'Unknown')
                    summary = result.get('summary', {})
                    
                    report_lines.extend([
                        f"Top-K: {top_k}",
                        f"  Average Retrieval Time: {summary.get('avg_retrieval_time', 0):.3f}s",
                        f"  Median Retrieval Time: {summary.get('median_retrieval_time', 0):.3f}s",
                        f"  Min/Max Time: {summary.get('min_retrieval_time', 0):.3f}s / {summary.get('max_retrieval_time', 0):.3f}s",
                        ""
                    ])
            
            elif test_name == 'end_to_end_rag':
                for result in benchmark.get('results', []):
                    temperature = result.get('temperature', 'Unknown')
                    summary = result.get('summary', {})
                    
                    report_lines.extend([
                        f"Temperature: {temperature}",
                        f"  Average Total Time: {summary.get('avg_total_time', 0):.3f}s",
                        f"  Average Retrieval Time: {summary.get('avg_retrieval_time', 0):.3f}s",
                        f"  Average Generation Time: {summary.get('avg_generation_time', 0):.3f}s",
                        ""
                    ])
            
            elif test_name == 'memory_usage':
                mem_results = benchmark.get('results', {})
                report_lines.extend([
                    f"Baseline Memory: {mem_results.get('baseline_memory_mb', 0):.1f} MB",
                    f"With Embedding Model: {mem_results.get('with_embedding_model_mb', 0):.1f} MB",
                    f"Model Overhead: {mem_results.get('embedding_model_overhead_mb', 0):.1f} MB",
                    ""
                ])
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 60,
            "End of Report",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
    def save_report(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save a human-readable report to file.
        
        Args:
            results: Benchmark results dictionary
            filename: Output filename (without extension)
            
        Returns:
            Path to saved report file
        """
        report = self.generate_report(results)
        output_path = self.output_dir / f"{filename}_report.txt"
        
        try:
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Benchmark report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise 