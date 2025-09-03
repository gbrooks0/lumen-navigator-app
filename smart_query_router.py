#!/usr/bin/env python3
"""
Smart Query Router for Dual Index RAG System - FIXED VERSION

This module provides intelligent routing between different embedding models
with Pydantic v2 compatibility fixes for vector store loading.
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

# LangChain imports with error handling
try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain.docstore.document import Document
    from langchain_core.documents import Document as CoreDocument
except ImportError as e:
    logging.error(f"LangChain import error: {e}")
    raise

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths
DB_BASE_DIR = "indexes"
OPENAI_DB_DIR = os.path.join(DB_BASE_DIR, "openai_index")
GOOGLE_DB_DIR = os.path.join(DB_BASE_DIR, "google_index")
METADATA_DIR = "metadata"
PERFORMANCE_DIR = "performance_metrics"

# Routing configuration
ROUTING_CONFIG = {
    "query_patterns": {
        "technical": {
            "keywords": ["code", "programming", "api", "algorithm", "software", "development", 
                        "python", "javascript", "database", "server", "framework", "library"],
            "preferred_provider": "openai",
            "confidence_boost": 0.15
        },
        "legal": {
            "keywords": ["regulation", "compliance", "policy", "framework", "guidance", "law",
                        "legal", "statute", "act", "regulation", "safeguarding", "inspection"],
            "preferred_provider": "google",
            "confidence_boost": 0.10
        },
        "general": {
            "keywords": ["what", "how", "why", "explain", "describe", "tell", "help", "show"],
            "preferred_provider": "openai",
            "confidence_boost": 0.05
        },
        "analytical": {
            "keywords": ["analyze", "compare", "evaluate", "assess", "review", "examine", "study"],
            "preferred_provider": "openai",
            "confidence_boost": 0.12
        }
    },
    "fallback_strategy": "performance_based",
    "default_provider": "openai",
    "min_confidence_threshold": 0.3,
    "performance_window_hours": 24,
    "max_response_time_seconds": 10.0
}

# Vision and LLM Configuration
VISION_CONFIG = {
    "models": {
        "openai_vision": {"model": "gpt-4o", "speed_tier": "slow", "quality_tier": "high"},
        "openai_vision_mini": {"model": "gpt-4o-mini", "speed_tier": "fast", "quality_tier": "medium"},
        "google_vision": {"model": "gemini-1.5-pro", "speed_tier": "slow", "quality_tier": "high"},
        "google_vision_flash": {"model": "gemini-1.5-flash", "speed_tier": "fast", "quality_tier": "medium"}
    },
    "routing_rules": {
        "speed": ["openai_vision_mini", "google_vision_flash"],
        "balanced": ["google_vision_flash", "openai_vision_mini", "google_vision"],
        "quality": ["openai_vision", "google_vision"]
    },
    "default_strategy": "balanced"
}

LLM_CONFIG = {
    "models": {
        "gpt_4o": {"speed_tier": "slow", "quality_tier": "high"},
        "gpt_4o_mini": {"speed_tier": "fast", "quality_tier": "medium"},
        "gemini_1_5_pro": {"speed_tier": "medium", "quality_tier": "high"},
        "gemini_1_5_flash": {"speed_tier": "fast", "quality_tier": "medium"}
    }
}

PERFORMANCE_METRICS = {
    "response_time_weight": 0.3,
    "result_quality_weight": 0.4,
    "success_rate_weight": 0.3,
    "decay_factor": 0.95
}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC COMPATIBILITY HELPERS
# =============================================================================

def safe_vector_store_load(db_path: str, embedding_model, max_retries: int = 3):
    """
    Safely load vector store with Pydantic compatibility handling.
    
    Args:
        db_path: Path to the vector store
        embedding_model: Embedding model instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        Loaded FAISS vector store or None if failed
    """
    for attempt in range(max_retries):
        try:
            # Method 1: Standard loading with dangerous deserialization
            vector_store = FAISS.load_local(
                db_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Successfully loaded vector store from {db_path} (attempt {attempt + 1})")
            return vector_store
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {db_path}: {str(e)}")
            
            # Method 2: Try with different loading parameters
            if attempt == 1:
                try:
                    # Sometimes specifying normalize_L2 helps
                    vector_store = FAISS.load_local(
                        db_path, 
                        embedding_model,
                        allow_dangerous_deserialization=True,
                        normalize_L2=False
                    )
                    logger.info(f"Successfully loaded vector store with normalize_L2=False")
                    return vector_store
                except Exception as e2:
                    logger.warning(f"Method 2 also failed: {str(e2)}")
            
            # Method 3: Try manual loading on final attempt
            if attempt == max_retries - 1:
                try:
                    return manual_vector_store_load(db_path, embedding_model)
                except Exception as e3:
                    logger.error(f"Manual loading also failed: {str(e3)}")
            
            time.sleep(1)  # Brief pause between retries
    
    logger.error(f"All attempts to load vector store from {db_path} failed")
    return None

def manual_vector_store_load(db_path: str, embedding_model):
    """
    Manually load vector store by handling files directly.
    This bypasses some Pydantic issues by reconstructing the store.
    """
    try:
        import pickle
        
        # Check if required files exist
        index_file = os.path.join(db_path, "index.faiss")
        pkl_file = os.path.join(db_path, "index.pkl")
        
        if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
            raise FileNotFoundError(f"Required FAISS files not found in {db_path}")
        
        # Load the pickle file manually and handle Pydantic issues
        with open(pkl_file, 'rb') as f:
            store_data = pickle.load(f)
        
        # Reconstruct FAISS store
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import faiss
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        
        # Create new docstore if needed
        if hasattr(store_data, 'docstore'):
            docstore = store_data.docstore
        else:
            docstore = InMemoryDocstore({})
        
        # Get index to docstore mapping
        if hasattr(store_data, 'index_to_docstore_id'):
            index_to_docstore_id = store_data.index_to_docstore_id
        else:
            index_to_docstore_id = {}
        
        # Reconstruct FAISS vector store
        vector_store = FAISS(
            embedding_function=embedding_model.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        logger.info(f"Successfully manually loaded vector store from {db_path}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Manual vector store loading failed: {str(e)}")
        return None

# =============================================================================
# PERFORMANCE TRACKER (Updated with better error handling)
# =============================================================================

class PerformanceTracker:
    """Tracks and analyzes performance metrics for each embedding provider."""
    
    def __init__(self):
        self.metrics_file = Path(PERFORMANCE_DIR) / "routing_metrics.json"
        self.ensure_performance_dir()
        self.load_metrics()
    
    def ensure_performance_dir(self):
        """Create performance directory if it doesn't exist."""
        try:
            Path(PERFORMANCE_DIR).mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create performance directory: {e}")
    
    def load_metrics(self):
        """Load existing performance metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                logger.info("Loaded existing performance metrics")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
                self.metrics = self._initialize_metrics()
        else:
            self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metrics structure."""
        return {
            "providers": {},
            "query_patterns": {},
            "last_updated": datetime.now().isoformat(),
            "total_queries": 0
        }
    
    def record_query(self, provider: str, query: str, response_time: float, 
                    result_count: int, pattern: str, success: bool = True):
        """Record query performance metrics with error handling."""
        try:
            if provider not in self.metrics["providers"]:
                self.metrics["providers"][provider] = {
                    "total_queries": 0,
                    "total_response_time": 0,
                    "success_count": 0,
                    "average_results": 0,
                    "pattern_performance": {},
                    "recent_queries": []
                }
            
            provider_metrics = self.metrics["providers"][provider]
            
            # Update provider metrics
            provider_metrics["total_queries"] += 1
            provider_metrics["total_response_time"] += response_time
            if success:
                provider_metrics["success_count"] += 1
            
            # Update pattern-specific metrics
            if pattern not in provider_metrics["pattern_performance"]:
                provider_metrics["pattern_performance"][pattern] = {
                    "queries": 0,
                    "avg_response_time": 0,
                    "success_rate": 0
                }
            
            pattern_metrics = provider_metrics["pattern_performance"][pattern]
            pattern_metrics["queries"] += 1
            
            # Update recent queries (keep last 50 to reduce memory usage)
            recent_query = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:50],  # Truncate for storage
                "response_time": response_time,
                "result_count": result_count,
                "pattern": pattern,
                "success": success
            }
            
            provider_metrics["recent_queries"].append(recent_query)
            if len(provider_metrics["recent_queries"]) > 50:
                provider_metrics["recent_queries"] = provider_metrics["recent_queries"][-50:]
            
            # Update global metrics
            self.metrics["total_queries"] += 1
            self.metrics["last_updated"] = datetime.now().isoformat()
            
            # Save metrics periodically
            if self.metrics["total_queries"] % 10 == 0:
                self.save_metrics()
                
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
    
    def get_provider_performance(self, provider: str, hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for a provider within time window."""
        try:
            if provider not in self.metrics["providers"]:
                return {"score": 0.0, "response_time": 10.0, "success_rate": 0.0, "total_queries": 0}
            
            provider_metrics = self.metrics["providers"][provider]
            
            # Calculate average response time
            if provider_metrics["total_queries"] > 0:
                avg_response_time = provider_metrics["total_response_time"] / provider_metrics["total_queries"]
                success_rate = provider_metrics["success_count"] / provider_metrics["total_queries"]
            else:
                avg_response_time = 10.0
                success_rate = 0.0
            
            # Calculate composite performance score
            response_time_score = max(0, (10.0 - avg_response_time) / 10.0)
            performance_score = (
                response_time_score * PERFORMANCE_METRICS["response_time_weight"] +
                success_rate * PERFORMANCE_METRICS["success_rate_weight"] +
                0.5 * PERFORMANCE_METRICS["result_quality_weight"]
            )
            
            return {
                "score": performance_score,
                "response_time": avg_response_time,
                "success_rate": success_rate,
                "total_queries": provider_metrics["total_queries"]
            }
        except Exception as e:
            logger.error(f"Error getting provider performance: {e}")
            return {"score": 0.0, "response_time": 10.0, "success_rate": 0.0, "total_queries": 0}
    
    def get_best_provider(self, pattern: str = None) -> str:
        """Get the best performing provider overall or for a specific pattern."""
        try:
            if not self.metrics["providers"]:
                return ROUTING_CONFIG["default_provider"]
            
            best_provider = ROUTING_CONFIG["default_provider"]
            best_score = 0.0
            
            for provider in self.metrics["providers"]:
                performance = self.get_provider_performance(provider)
                score = performance["score"]
                
                # Boost score for pattern-specific performance
                if pattern and provider in self.metrics["providers"]:
                    provider_metrics = self.metrics["providers"][provider]
                    if pattern in provider_metrics["pattern_performance"]:
                        pattern_perf = provider_metrics["pattern_performance"][pattern]
                        if pattern_perf["queries"] > 0:
                            score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_provider = provider
            
            return best_provider
        except Exception as e:
            logger.error(f"Error getting best provider: {e}")
            return ROUTING_CONFIG["default_provider"]
    
    def save_metrics(self):
        """Save metrics to file with error handling."""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")

# =============================================================================
# VISION PERFORMANCE TRACKER
# =============================================================================

class VisionPerformanceTracker(PerformanceTracker):
    """Extended performance tracker for vision and LLM models."""
    
    def __init__(self):
        super().__init__()
        self.vision_metrics = {
            "vision_models": {},
            "llm_models": {},
            "speed_vs_quality": {}
        }
    
    def record_vision_query(self, vision_model: str, llm_model: str, 
                           image_size_kb: int, query_complexity: str,
                           vision_time: float, llm_time: float, total_time: float):
        """Record performance metrics for vision + LLM pipeline."""
        try:
            # Vision model tracking
            if vision_model not in self.vision_metrics["vision_models"]:
                self.vision_metrics["vision_models"][vision_model] = {
                    "total_queries": 0, "total_time": 0
                }
            
            vision_stats = self.vision_metrics["vision_models"][vision_model]
            vision_stats["total_queries"] += 1
            vision_stats["total_time"] += vision_time
            
            # LLM model tracking
            if llm_model not in self.vision_metrics["llm_models"]:
                self.vision_metrics["llm_models"][llm_model] = {
                    "total_queries": 0, "total_time": 0
                }
            
            llm_stats = self.vision_metrics["llm_models"][llm_model]
            llm_stats["total_queries"] += 1
            llm_stats["total_time"] += llm_time
        except Exception as e:
            logger.error(f"Error recording vision query: {e}")
    
    def get_best_vision_model(self, priority: str = "balanced") -> str:
        """Get best vision model based on priority."""
        try:
            if priority not in VISION_CONFIG["routing_rules"]:
                priority = VISION_CONFIG["default_strategy"]
            
            priority_models = VISION_CONFIG["routing_rules"][priority]
            
            # Check performance metrics if available
            if self.vision_metrics["vision_models"]:
                best_model = None
                best_avg_time = float('inf')
                
                for model in priority_models:
                    if model in self.vision_metrics["vision_models"]:
                        stats = self.vision_metrics["vision_models"][model]
                        if stats["total_queries"] > 0:
                            avg_time = stats["total_time"] / stats["total_queries"]
                            if avg_time < best_avg_time:
                                best_avg_time = avg_time
                                best_model = model
                
                if best_model:
                    return best_model
            
            # Fallback to first available model in priority list
            return priority_models[0] if priority_models else "openai_vision_mini"
        except Exception as e:
            logger.error(f"Error getting best vision model: {e}")
            return "openai_vision_mini"
    
    def get_best_llm_model(self, complexity: str = "medium") -> str:
        """Get best LLM model based on complexity."""
        try:
            if complexity == "low":
                candidates = ["gpt_4o_mini", "gemini_1_5_flash"]
            elif complexity == "high":
                candidates = ["gpt_4o", "gemini_1_5_pro"] 
            else:  # medium
                candidates = ["gpt_4o_mini", "gemini_1_5_flash", "gemini_1_5_pro"]
            
            # Check performance metrics
            if self.vision_metrics["llm_models"]:
                best_model = None
                best_time = float('inf')
                
                for model in candidates:
                    if model in self.vision_metrics["llm_models"]:
                        stats = self.vision_metrics["llm_models"][model]
                        if stats["total_queries"] > 0:
                            avg_time = stats["total_time"] / stats["total_queries"]
                            if avg_time < best_time:
                                best_time = avg_time
                                best_model = model
                
                if best_model:
                    return best_model
            
            return candidates[0] if candidates else "gpt_4o_mini"
        except Exception as e:
            logger.error(f"Error getting best LLM model: {e}")
            return "gpt_4o_mini"

# =============================================================================
# QUERY ANALYZER
# =============================================================================

class QueryAnalyzer:
    """Enhanced query analyzer that detects authority requirements and adaptive retrieval needs."""
    
    def __init__(self):
        # Add temporal detection patterns
        self.temporal_patterns = [
            r'\blatest\b', r'\brecent\b', r'\bcurrent\b', r'\bnew\b', r'\bupdated\b',
            r'\b202[4-9]\b', r'\bthis year\b', r'\blast year\b', r'\btoday\b',
            r'\bchanges?\s+(?:to|in|for)\b', r'\brevised?\b', r'\bamended?\b'
        ]

        # Comparison detection patterns
        self.comparison_patterns = [
            r'\bcompare\b', r'\bversus\b', r'\bvs\.?\b', r'\bdifference[s]?\s+between\b',
            r'\bcontrast\b', r'\bbetter\s+than\b', r'\bworse\s+than\b',
            r'\bkey\s+differences?\b', r'\bhow\s+(?:do|does)\s+.*\s+differ\b',
            r'\bsimilarit(?:ies|y)\s+(?:and\s+)?differences?\b'
        ]
        
        # Your existing patterns from ROUTING_CONFIG
        self.patterns = {
            "technical": {
                "keywords": ["code", "programming", "api", "algorithm", "software", "development"],
                "preferred_provider": "openai",
                "confidence_boost": 0.15
            },
            "legal": {
                "keywords": ["regulation", "compliance", "policy", "framework", "guidance", "law"],
                "preferred_provider": "google", 
                "confidence_boost": 0.10
            },
            "general": {
                "keywords": ["what", "how", "why", "explain", "describe", "tell", "help", "show"],
                "preferred_provider": "openai",
                "confidence_boost": 0.05
            }
        }
        
        # NEW: Add retrieval sizing patterns
        self.retrieval_patterns = {
            "single_fact": {
                "keywords": ["what is", "define", "who is", "when was", "where is"],
                "optimal_k": 3,
                "max_k": 5
            },
            "specific_list": {
                "keywords": ["what are the", "list the", "9 quality standards", "how many"],
                "optimal_k": 2,
                "max_k": 4
            },
            "comparison": {
                "keywords": ["compare", "difference between", "versus", "contrast"],
                "optimal_k": 6,
                "max_k": 8
            },
            "comprehensive": {
                "keywords": ["explain", "describe", "how to", "process", "procedure"],
                "optimal_k": 5,
                "max_k": 8
            }
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis with temporal awareness, comparison detection, and adaptive retrieval sizing."""
        try:
            query_lower = query.lower()
            
            # ISSUE #1 FIX: Detect temporal queries and add knowledge currency warning
            is_temporal = any(re.search(pattern, query_lower) for pattern in self.temporal_patterns)
            
            # ISSUE #2 FIX: Detect comparison queries that need more documents
            is_comparison = any(re.search(pattern, query_lower) for pattern in self.comparison_patterns)
            
            # Existing pattern analysis
            pattern_scores = {}
            for pattern_name, pattern_config in self.patterns.items():
                score = 0.0
                keywords_found = []
                
                for keyword in pattern_config["keywords"]:
                    if keyword in query_lower:
                        score += 1.0
                        keywords_found.append(keyword)
                
                if len(pattern_config["keywords"]) > 0:
                    score = score / len(pattern_config["keywords"])
                
                pattern_scores[pattern_name] = {
                    "score": score,
                    "keywords_found": keywords_found,
                    "preferred_provider": pattern_config["preferred_provider"],
                    "confidence_boost": pattern_config.get("confidence_boost", 0.0)
                }
            
            # Determine best pattern match
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1]["score"])
            pattern_name = best_pattern[0]
            pattern_data = best_pattern[1]
            
            # Calculate confidence
            base_confidence = min(pattern_data["score"], 1.0)
            confidence = base_confidence + pattern_data["confidence_boost"]
            confidence = min(confidence, 1.0)
            
            # ISSUE #2 FIX: Determine optimal k with comparison boost
            optimal_k, max_k = self._determine_optimal_k(query_lower, is_comparison)
            
            return {
                "query": query,
                "best_pattern": pattern_name,
                "pattern_confidence": confidence,
                "preferred_provider": pattern_data["preferred_provider"],
                "keywords_found": pattern_data["keywords_found"],
                "query_length": len(query.split()),
                "all_pattern_scores": pattern_scores,
                "optimal_k": optimal_k,
                "max_k": max_k,
                "is_temporal": is_temporal,  # NEW: Temporal detection
                "is_comparison": is_comparison,  # NEW: Comparison detection
                "needs_currency_disclaimer": is_temporal,  # NEW: Currency warning flag
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "query": query,
                "best_pattern": "general",
                "pattern_confidence": 0.5,
                "preferred_provider": "openai",
                "keywords_found": [],
                "query_length": len(query.split()),
                "optimal_k": 5,
                "max_k": 7,
                "is_temporal": False,
                "is_comparison": False,
                "needs_currency_disclaimer": False,
                "error": str(e)
            }

    def _determine_optimal_k(self, query_lower: str, is_comparison: bool = False) -> tuple:
        """Determine optimal number of documents based on query type with comparison detection."""
        
        # ISSUE #2 FIX: Comparison queries need more documents
        if is_comparison:
            return 7, 10  # Increased for proper comparative analysis
        
        # Check retrieval patterns
        for pattern_name, pattern_config in self.retrieval_patterns.items():
            for keyword in pattern_config["keywords"]:
                if keyword in query_lower:
                    return pattern_config["optimal_k"], pattern_config["max_k"]
        
        # Query length heuristics
        word_count = len(query_lower.split())
        if word_count <= 5:
            return 3, 5
        elif word_count <= 10:
            return 5, 7
        else:
            return 7, 10
    
    def _determine_retrieval_strategy(self, query_lower: str, analysis: Dict) -> str:
        """Determine retrieval strategy for efficiency."""
        if analysis.get("requires_authoritative") and analysis.get("requires_comprehensive"):
            return "authoritative_then_expand"
        elif "what are the" in query_lower and any(x in query_lower for x in ["standards", "requirements", "regulations"]):
            return "precision_first"
        elif analysis.get("query_complexity") == "high":
            return "comprehensive_search"
        else:
            return "standard_adaptive"


# =============================================================================
# ENHANCED SMART ROUTER
# =============================================================================

class SmartRouter:
    """Main router class with enhanced error handling and Pydantic compatibility."""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.query_analyzer = QueryAnalyzer()
        self.embedding_models = {}
        self.vector_stores = {}
        self.vision_tracker = VisionPerformanceTracker()
        self.performance_mode = "balanced"
        
        # Load models and stores with error handling
        self.load_embedding_models()
        
        # ADD: Validate embedding consistency before loading vector stores
        validation = self.validate_embedding_consistency()
        inconsistent_models = [p for p, v in validation.items() if v.get("status") == "dimension_mismatch"]
        
        if inconsistent_models:
            logger.error(f"Embedding dimension mismatches detected: {inconsistent_models}")
            logger.error("Vector stores may fail to load. Consider rebuilding indexes with consistent models.")
        
        self.load_vector_stores()
    
    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'quality'."""
        if mode in ["speed", "balanced", "quality"]:
            self.performance_mode = mode
            logger.info(f"Performance mode set to: {mode}")
        else:
            logger.warning(f"Invalid mode {mode}. Valid modes: speed, balanced, quality")
    
    def load_embedding_models(self):
        """Load embedding models with improved error handling."""
        model_info_file = Path(METADATA_DIR) / "embedding_models.json"
        
        # Try to load from metadata file first
        if model_info_file.exists():
            try:
                with open(model_info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                for provider, info in model_info.items():
                    if info.get("status") == "active":
                        try:
                            if provider == "openai":
                                # Use the model from metadata but default to text-embedding-3-small
                                model_name = info.get("model", "text-embedding-3-small")
                                # FORCE CONSISTENCY: Override large model if found
                                if model_name == "text-embedding-3-large":
                                    model_name = "text-embedding-3-small"
                                    logger.warning(f"Overriding {provider} model to text-embedding-3-small for consistency")
                                
                                self.embedding_models[provider] = OpenAIEmbeddings(
                                    model=model_name,
                                    show_progress_bar=False
                                )
                            elif provider == "google":
                                self.embedding_models[provider] = GoogleGenerativeAIEmbeddings(
                                    model=info.get("model", "models/text-embedding-004")
                                )
                            logger.info(f"✅ Loaded {provider} embedding model")
                        except Exception as e:
                            logger.error(f"⚠ Failed to load {provider} embedding model: {e}")
            except Exception as e:
                logger.warning(f"Failed to load model info from {model_info_file}: {e}")
        
        # Fallback to default models if none loaded
        if not self.embedding_models:
            logger.warning("No embedding models loaded from metadata, trying defaults...")
            self._load_default_models()
    
    def _load_default_models(self):
        """Load default embedding models as fallback."""
        try:
            # Try OpenAI first
            try:
                self.embedding_models["openai"] = OpenAIEmbeddings(
                    model="text-embedding-3-small",  # CHANGED from text-embedding-3-large
                    show_progress_bar=False
                )
                logger.info("✅ Loaded default OpenAI embedding model")
            except Exception as e:
                logger.warning(f"Failed to load default OpenAI model: {e}")
            
            # Try Google - this stays the same
            try:
                self.embedding_models["google"] = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004"  # This can stay as is
                )
                logger.info("✅ Loaded default Google embedding model")
            except Exception as e:
                logger.warning(f"Failed to load default Google model: {e}")
                
        except Exception as e:
            logger.error(f"Failed to load any default embedding models: {e}")
    
    def load_vector_stores(self):
        """Load FAISS vector stores with automatic rebuild capability."""
        store_paths = {
            "openai": OPENAI_DB_DIR,
            "google": GOOGLE_DB_DIR
        }
        
        cached_documents = None
        
        for provider, db_path in store_paths.items():
            if provider in self.embedding_models:
                if os.path.exists(db_path):
                    vector_store = safe_vector_store_load(db_path, self.embedding_models[provider])
                    
                    if vector_store is not None:
                        # Validate dimensions match
                        try:
                            test_embedding = self.embedding_models[provider].embed_query("test")
                            expected_dims = len(test_embedding)
                            
                            # Check if vector store dimensions match
                            if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'd'):
                                store_dims = vector_store.index.d
                                if store_dims != expected_dims:
                                    logger.error(f"Dimension mismatch for {provider}: store has {store_dims}, model expects {expected_dims}")
                                    logger.info(f"Rebuilding {provider} store due to dimension mismatch")
                                    # Trigger rebuild
                                    if cached_documents is None:
                                        cached_documents = self._load_from_document_cache()
                                    self._rebuild_store_from_cache(provider, db_path, cached_documents)
                                    continue
                            
                            self.vector_stores[provider] = vector_store
                            logger.info(f"Successfully loaded {provider} vector store")
                            
                        except Exception as validation_error:
                            logger.error(f"Error validating {provider} store: {validation_error}")
                            self.vector_stores[provider] = vector_store  # Use anyway
                            
                    else:
                        # Load documents once for all rebuilds
                        if cached_documents is None:
                            cached_documents = self._load_from_document_cache()
                        self._rebuild_store_from_cache(provider, db_path, cached_documents)
                else:
                    if cached_documents is None:
                        cached_documents = self._load_from_document_cache()
                    self._rebuild_store_from_cache(provider, db_path, cached_documents)

    def validate_embedding_consistency(self) -> Dict[str, Any]:
        """Validate that embedding models match existing vector store dimensions."""
        validation_results = {}
        
        for provider, model in self.embedding_models.items():
            try:
                # Test embedding to get actual dimensions
                test_embedding = model.embed_query("test")
                actual_dims = len(test_embedding)
                
                # Check against expected dimensions for consistency
                expected_dims = {
                    "openai": 1536,  # text-embedding-3-small
                    "google": 768    # text-embedding-004
                }
                
                expected = expected_dims.get(provider, actual_dims)
                is_consistent = actual_dims == expected
                
                validation_results[provider] = {
                    "model": getattr(model, 'model', 'unknown'),
                    "actual_dimensions": actual_dims,
                    "expected_dimensions": expected,
                    "is_consistent": is_consistent,
                    "status": "valid" if is_consistent else "dimension_mismatch"
                }
                
                if not is_consistent:
                    logger.error(f"Dimension mismatch for {provider}: got {actual_dims}, expected {expected}")
                
            except Exception as e:
                validation_results[provider] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return validation_results

    def _rebuild_store_from_cache(self, provider: str, db_path: str, cached_documents: List[Document] = None):
        """Rebuild vector store from document cache."""
        try:
            # Remove corrupted store
            if os.path.exists(db_path):
                import shutil
                backup_path = f"{db_path}_corrupted_{int(time.time())}"
                logger.info(f"Moving corrupted store to {backup_path}")
                shutil.move(db_path, backup_path)
            
            # Use provided documents or load from cache
            if cached_documents is None:
                documents = self._load_from_document_cache()
            else:
                documents = cached_documents
            
            if not documents:
                logger.error(f"No documents found in cache to rebuild {provider} store")
                return
            
            logger.info(f"Rebuilding {provider} vector store with {len(documents)} documents")
            
            # Create vector store with retry logic
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            vector_store = self._create_vector_store_with_retry(provider, texts, metadatas)
            
            # Save the new store
            os.makedirs(db_path, exist_ok=True)
            vector_store.save_local(db_path)
            
            self.vector_stores[provider] = vector_store
            logger.info(f"Successfully rebuilt and loaded {provider} vector store")
            
        except Exception as e:
            logger.error(f"Failed to rebuild {provider} store: {e}")
            # Continue without this provider rather than failing completely

    def _load_from_document_cache(self):
        """Load documents from document_cache directory."""
        documents = []
        cache_dir = "document_cache"
        
        if not os.path.exists(cache_dir):
            logger.error("document_cache directory not found")
            return documents
        
        try:
            cache_files = list(Path(cache_dir).glob("*.json"))
            logger.info(f"Loading documents from {len(cache_files)} cache files...")
            
            for cache_file in cache_files:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            content = item.get('content', item.get('page_content', ''))
                            if content:
                                documents.append(Document(
                                    page_content=content,
                                    metadata=item.get('metadata', {})
                                ))
            
            logger.info(f"Loaded {len(documents)} documents from cache")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading from document cache: {e}")
            return []

    def _create_vector_store_with_retry(self, provider: str, texts: List[str], metadatas: List[Dict], max_retries: int = 3):
        """Create vector store with retry logic for API rate limits."""
        for attempt in range(max_retries):
            try:
                if provider == "google":
                    # Add delay for Google to avoid rate limits
                    time.sleep(2)
                    
                vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_models[provider],
                    metadatas=metadatas
                )
                return vector_store
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    logger.warning(f"Rate limit hit for {provider}, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Vector store creation failed for {provider} (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
        
        raise Exception(f"Failed to create vector store for {provider} after {max_retries} attempts")

    def route_query(self, query: str, k: int = 5, force_provider: str = None) -> Dict[str, Any]:
        """Route query to optimal embedding provider and return results."""
        start_time = time.time()
        
        try:
            # Check if any vector stores are available
            if not self.vector_stores:
                return {
                    "success": False,
                    "error": "No vector stores available",
                    "documents": [],
                    "total_results": 0,
                    "response_time": time.time() - start_time,
                    "provider": "none",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Use adaptive k if k is None
            analysis = self.query_analyzer.analyze_query(query)
            if k is None:
                k = analysis.get("optimal_k", 5)
            
            # Select provider
            if force_provider:
                selected_provider = force_provider
            else:
                selected_provider = self._select_provider(analysis)
            
            # Execute query
            result = self._execute_query(selected_provider, query, k, start_time, analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in route_query: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "total_results": 0,
                "response_time": time.time() - start_time,
                "provider": "error",
                "timestamp": datetime.now().isoformat()
            }       


    def analyze_query_complexity(self, query: str, has_image: bool = False) -> str:
        """Analyze query complexity for model selection."""
        try:
            query_lower = query.lower()
        
            high_complexity_keywords = [
                "analyze", "compare", "evaluate", "assess", "detailed", "comprehensive",
                "complex", "multi-step", "reasoning", "critical", "safety", "risk"
            ]
        
            low_complexity_keywords = [
                "what", "who", "when", "where", "simple", "basic", "quick", "summary"
            ]
        
            high_score = sum(1 for keyword in high_complexity_keywords if keyword in query_lower)
            low_score = sum(1 for keyword in low_complexity_keywords if keyword in query_lower)
        
            query_length = len(query.split())
            complexity_boost = 1 if has_image else 0
            
            if high_score >= 2 or query_length > 30 or complexity_boost:
                return "high"
            elif low_score >= 2 and query_length < 10:
                return "low" 
            else:
                return "medium"
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return "medium"
    
    def enhanced_route_query(self, query: str, k: int = None, force_provider: str = None) -> Dict[str, Any]:
        """Enhanced routing with adaptive document retrieval."""
        start_time = time.time()
        
        try:
            # Enhanced query analysis
            analysis = self.query_analyzer.analyze_query(query)
            
            # Use adaptive k if not specified
            if k is None:
                k = analysis.get("optimal_k", 5)
            
            # Get retrieval strategy
            retrieval_strategy = analysis.get("retrieval_strategy", "standard_adaptive")
            
            # Select provider
            if force_provider:
                selected_provider = force_provider
            else:
                routing_decision = analysis["final_routing_decision"]
                selected_provider = routing_decision["provider_preference"]
            
            if selected_provider not in self.vector_stores:
                available_providers = list(self.vector_stores.keys())
                selected_provider = available_providers[0] if available_providers else None
            
            if not selected_provider:
                return self._create_error_response("No vector stores available", analysis, start_time)
            
            # Execute adaptive retrieval
            result = self._execute_adaptive_retrieval(
                selected_provider, query, k, start_time, analysis, retrieval_strategy
            )
            
            return result
            
        except Exception as e:
            return self._create_error_response(str(e), analysis if 'analysis' in locals() else {}, start_time)
    
    def route_multimodal_query(self, query: str, image_data: bytes = None, 
                              image_size_kb: int = None, k: int = 5) -> Dict[str, Any]:
        """Route multimodal query with optimal model selection."""
        start_time = time.time()
        
        try:
            # Analyze query complexity
            complexity = self.analyze_query_complexity(query, has_image=bool(image_data))
            
            # Route text retrieval (existing functionality)
            text_result = self.route_query(query, k=k)
            
            # Select vision model if image provided
            vision_model = None
            if image_data:
                vision_model = self.vision_tracker.get_best_vision_model(
                    priority=self.performance_mode
                )
            
            # Select LLM model based on complexity and mode
            llm_model = self.vision_tracker.get_best_llm_model(complexity=complexity)
            
            # Build response
            routing_result = {
                "text_routing": text_result,
                "vision_model": vision_model,
                "llm_model": llm_model,
                "performance_mode": self.performance_mode,
                "query_complexity": complexity,
                "timestamp": datetime.now().isoformat()
            }
            
            return routing_result
        except Exception as e:
            logger.error(f"Error in multimodal routing: {e}")
            return {
                "text_routing": {"success": False, "error": str(e), "documents": []},
                "vision_model": None,
                "llm_model": "gpt_4o_mini",
                "performance_mode": self.performance_mode,
                "query_complexity": "medium",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _select_provider(self, analysis: Dict[str, Any]) -> str:
        """Select the best provider based on analysis and performance."""
        try:
            pattern = analysis.get("best_pattern", "general")
            confidence = analysis.get("pattern_confidence", 0.0)
            preferred_provider = analysis.get("preferred_provider", ROUTING_CONFIG["default_provider"])
            
            # If confidence is high, use pattern preference
            if confidence >= ROUTING_CONFIG["min_confidence_threshold"]:
                # Check if preferred provider is available
                if preferred_provider in self.vector_stores:
                    return preferred_provider
            
            # Otherwise, use performance-based selection
            if ROUTING_CONFIG["fallback_strategy"] == "performance_based":
                return self.performance_tracker.get_best_provider(pattern)
            else:
                # Return first available provider
                available_providers = list(self.vector_stores.keys())
                if available_providers:
                    return available_providers[0]
                else:
                    return ROUTING_CONFIG["default_provider"]
        except Exception as e:
            logger.error(f"Error selecting provider: {e}")
            return ROUTING_CONFIG["default_provider"]
    
    def _get_fallback_provider(self, primary_provider: str) -> Optional[str]:
        """Get fallback provider different from primary."""
        try:
            available_providers = list(self.vector_stores.keys())
            fallback_candidates = [p for p in available_providers if p != primary_provider]
            
            if not fallback_candidates:
                return None
            
            if ROUTING_CONFIG["fallback_strategy"] == "performance_based":
                best_fallback = None
                best_score = -1
                for provider in fallback_candidates:
                    perf = self.performance_tracker.get_provider_performance(provider)
                    if perf["score"] > best_score:
                        best_score = perf["score"]
                        best_fallback = provider
                return best_fallback
            else:
                return fallback_candidates[0]
        except Exception as e:
            logger.error(f"Error getting fallback provider: {e}")
            return None
    
    def _execute_query(self, provider: str, query: str, k: int, start_time: float, 
                      analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query against specific provider with comprehensive error handling."""
        try:
            if provider not in self.vector_stores:
                raise Exception(f"Vector store not available for {provider}")
            
            # Perform similarity search
            search_start = time.time()
            docs = self.vector_stores[provider].similarity_search(query, k=k)
            search_time = time.time() - search_start
            
            total_time = time.time() - start_time
            
            # Record performance
            pattern = analysis.get("best_pattern", "unknown")
            self.performance_tracker.record_query(
                provider=provider,
                query=query,
                response_time=total_time,
                result_count=len(docs),
                pattern=pattern,
                success=True
            )
            
            return {
                "success": True,
                "provider": provider,
                "documents": docs,
                "total_results": len(docs),
                "response_time": total_time,
                "search_time": search_time,
                "analysis": analysis,
                "used_fallback": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            
            # Record failed performance
            pattern = analysis.get("best_pattern", "unknown")
            self.performance_tracker.record_query(
                provider=provider,
                query=query,
                response_time=total_time,
                result_count=0,
                pattern=pattern,
                success=False
            )
            
            logger.error(f"Query execution failed for {provider}: {e}")
            return {
                "success": False,
                "provider": provider,
                "error": str(e),
                "documents": [],
                "total_results": 0,
                "response_time": total_time,
                "analysis": analysis,
                "used_fallback": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_adaptive_retrieval(self, provider: str, query: str, k: int, 
                                   start_time: float, analysis: Dict[str, Any], 
                                   strategy: str) -> Dict[str, Any]:
        """Execute retrieval with adaptive strategies for efficiency."""
        try:
            vector_store = self.vector_stores[provider]
            
            if strategy == "precision_first":
                # For specific queries like "what are the 9 standards", try small k first
                return self._precision_first_retrieval(vector_store, provider, query, k, analysis, start_time)
            
            elif strategy == "authoritative_then_expand":
                # Get authoritative sources first, expand if needed
                return self._authoritative_then_expand_retrieval(vector_store, provider, query, k, analysis, start_time)
            
            elif strategy == "comprehensive_search":
                # Complex queries need broader search
                return self._comprehensive_retrieval(vector_store, provider, query, k, analysis, start_time)
            
            else:
                # Standard adaptive retrieval
                return self._standard_adaptive_retrieval(vector_store, provider, query, k, analysis, start_time)
                
        except Exception as e:
            return self._create_error_response(f"Adaptive retrieval failed: {str(e)}", analysis, start_time)


    def _precision_first_retrieval(self, vector_store, provider: str, query: str, 
                                  k: int, analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Precision-first retrieval for specific queries."""
        try:
            # Start with small retrieval
            initial_k = min(3, k)
            docs = vector_store.similarity_search(query, k=initial_k)
            
            # Check if we have high-authority results
            high_authority_docs = [
                doc for doc in docs 
                if doc.metadata.get('authority_level', 0) >= 0.8
            ]
            
            # If we have good authoritative results, we might be done
            if len(high_authority_docs) >= 1 and analysis.get("requires_authoritative"):
                # Check if the top result seems comprehensive
                top_doc = high_authority_docs[0]
                if (top_doc.metadata.get('is_primary_source', False) or 
                    len(top_doc.metadata.get('standards_covered', [])) > 0):
                    
                    # We likely have what we need, but get one more for context
                    final_k = min(initial_k + 1, k)
                    if final_k > len(docs):
                        docs = vector_store.similarity_search(query, k=final_k)
                    
                    return self._create_success_response(
                        docs[:final_k], provider, analysis, start_time, 
                        f"Precision retrieval: found authoritative source with k={final_k}"
                    )
            
            # Need more documents
            if k > initial_k:
                docs = vector_store.similarity_search(query, k=k)
            
            return self._create_success_response(docs, provider, analysis, start_time, "Standard retrieval after precision check")
            
        except Exception as e:
            return self._create_error_response(f"Precision retrieval failed: {str(e)}", analysis, start_time)


    def _authoritative_then_expand_retrieval(self, vector_store, provider: str, query: str,
                                           k: int, analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Get authoritative sources first, then expand if needed."""
        try:
            # Get more candidates for filtering
            search_k = min(k * 2, 20)
            candidates = vector_store.similarity_search_with_score(query, k=search_k)
            
            # Separate by authority level
            high_authority = []
            medium_authority = []
            other_docs = []
            
            for doc, score in candidates:
                authority = doc.metadata.get('authority_level', 0)
                if authority >= 0.8:
                    high_authority.append((doc, score))
                elif authority >= 0.6:
                    medium_authority.append((doc, score))
                else:
                    other_docs.append((doc, score))
            
            # Build result set prioritizing authority
            final_docs = []
            
            # Add high authority docs first (up to 60% of k)
            high_authority_limit = max(1, int(k * 0.6))
            final_docs.extend([doc for doc, _ in high_authority[:high_authority_limit]])
            
            # Fill remaining with medium authority
            remaining = k - len(final_docs)
            if remaining > 0:
                final_docs.extend([doc for doc, _ in medium_authority[:remaining]])
            
            # Fill any remaining with other docs
            remaining = k - len(final_docs)
            if remaining > 0:
                final_docs.extend([doc for doc, _ in other_docs[:remaining]])
            
            return self._create_success_response(
                final_docs, provider, analysis, start_time,
                f"Authority-first retrieval: {len([d for d in final_docs if d.metadata.get('authority_level', 0) >= 0.8])} high-authority docs"
            )
            
        except Exception as e:
            return self._create_error_response(f"Authoritative retrieval failed: {str(e)}", analysis, start_time)


    def _comprehensive_retrieval(self, vector_store, provider: str, query: str,
                                k: int, analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Comprehensive retrieval for complex queries."""
        try:
            # Use the full k for complex queries, but ensure diversity
            docs = vector_store.similarity_search(query, k=k)
            
            # Apply diversity filtering to avoid too many chunks from same source
            diverse_docs = self._apply_diversity_filter(docs, max_per_source=3)
            
            return self._create_success_response(
                diverse_docs, provider, analysis, start_time, 
                f"Comprehensive retrieval with diversity filtering"
            )
            
        except Exception as e:
            return self._create_error_response(f"Comprehensive retrieval failed: {str(e)}", analysis, start_time)


    def _standard_adaptive_retrieval(self, vector_store, provider: str, query: str,
                                   k: int, analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Standard adaptive retrieval with smart k adjustment."""
        try:
            docs = vector_store.similarity_search(query, k=k)
            
            # For simple queries, check if we can reduce the result set
            if k > 3 and analysis.get("query_complexity", "medium") == "low":
                # Check if first few results are highly relevant and authoritative
                top_docs = docs[:3]
                if any(doc.metadata.get('authority_level', 0) >= 0.8 for doc in top_docs):
                    # We might have sufficient results with fewer docs
                    docs = docs[:min(k, 4)]
            
            return self._create_success_response(docs, provider, analysis, start_time, "Standard adaptive retrieval")
            
        except Exception as e:
            return self._create_error_response(f"Standard retrieval failed: {str(e)}", analysis, start_time)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all providers."""
        try:
            summary = {
                "providers": {},
                "total_queries": self.performance_tracker.metrics.get("total_queries", 0),
                "available_providers": list(self.vector_stores.keys()),
                "loaded_embedding_models": list(self.embedding_models.keys()),
                "last_updated": self.performance_tracker.metrics.get("last_updated"),
                "status": "healthy" if self.vector_stores else "no_vector_stores"
            }
            
            for provider in self.vector_stores.keys():
                summary["providers"][provider] = self.performance_tracker.get_provider_performance(provider)
            
            return summary
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                "providers": {},
                "total_queries": 0,
                "available_providers": [],
                "loaded_embedding_models": [],
                "last_updated": None,
                "status": "error",
                "error": str(e)
            }
    
    def test_routing(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test routing with sample queries for validation."""
        try:
            results = []
            
            for query in test_queries:
                try:
                    analysis = self.query_analyzer.analyze_query(query)
                    result = {
                        "query": query,
                        "analysis": analysis,
                        "recommended_provider": self._select_provider(analysis)
                    }
                    results.append(result)
                except Exception as e:
                    results.append({
                        "query": query,
                        "error": str(e),
                        "recommended_provider": ROUTING_CONFIG["default_provider"]
                    })
            
            return {"test_results": results, "total_tests": len(test_queries)}
        except Exception as e:
            logger.error(f"Error in test routing: {e}")
            return {"test_results": [], "total_tests": 0, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the routing system."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check embedding models
            health_status["components"]["embedding_models"] = {
                "status": "healthy" if self.embedding_models else "no_models",
                "count": len(self.embedding_models),
                "available": list(self.embedding_models.keys())
            }
            
            # Check vector stores
            health_status["components"]["vector_stores"] = {
                "status": "healthy" if self.vector_stores else "no_stores",
                "count": len(self.vector_stores),
                "available": list(self.vector_stores.keys())
            }
            
            # Check performance tracker
            try:
                perf_summary = self.get_performance_summary()
                health_status["components"]["performance_tracker"] = {
                    "status": "healthy",
                    "total_queries": perf_summary.get("total_queries", 0)
                }
            except Exception as e:
                health_status["components"]["performance_tracker"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Test basic query analysis
            try:
                test_analysis = self.query_analyzer.analyze_query("test query")
                health_status["components"]["query_analyzer"] = {
                    "status": "healthy"
                }
            except Exception as e:
                health_status["components"]["query_analyzer"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Determine overall status
            component_statuses = [comp["status"] for comp in health_status["components"].values()]
            if "error" in component_statuses:
                health_status["overall_status"] = "degraded"
            elif not self.vector_stores:
                health_status["overall_status"] = "limited"
            
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
        
        return health_status

    def _apply_diversity_filter(self, docs: List, max_per_source: int = 3) -> List:
        """Apply diversity filtering to avoid too many chunks from same source."""
        source_counts = {}
        filtered_docs = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            count = source_counts.get(source, 0)
            
            if count < max_per_source:
                filtered_docs.append(doc)
                source_counts[source] = count + 1
        
        return filtered_docs


    def _create_success_response(self, docs: List, provider: str, analysis: Dict, 
                               start_time: float, retrieval_note: str) -> Dict[str, Any]:
        """Create a standardized success response."""
        return {
            "success": True,
            "provider": provider,
            "documents": docs,
            "total_results": len(docs),
            "response_time": time.time() - start_time,
            "analysis": analysis,
            "retrieval_strategy": retrieval_note,
            "authority_filtered": analysis.get("requires_authoritative", False),
            "timestamp": datetime.now().isoformat()
        }


    def _create_error_response(self, error_msg: str, analysis: Dict, start_time: float) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "success": False,
            "error": error_msg,
            "documents": [],
            "response_time": time.time() - start_time,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_smart_router() -> SmartRouter:
    """Create and return a configured SmartRouter instance with error handling."""
    try:
        return SmartRouter()
    except Exception as e:
        logger.error(f"Failed to create SmartRouter: {e}")
        raise

def quick_search(query: str, k: int = 5, provider: str = None) -> List[Document]:
    """Quick search function for simple use cases with error handling."""
    try:
        router = create_smart_router()
        result = router.route_query(query, k=k, force_provider=provider)
        
        if result["success"]:
            return result["documents"]
        else:
            logger.error(f"Search failed: {result.get('error')}")
            return []
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        return []

def diagnose_vector_stores() -> Dict[str, Any]:
    """Diagnostic function to check vector store health."""
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "directories": {},
        "files": {},
        "recommendations": []
    }
    
    # Check directories
    for name, path in [("openai", OPENAI_DB_DIR), ("google", GOOGLE_DB_DIR)]:
        diagnosis["directories"][name] = {
            "path": path,
            "exists": os.path.exists(path),
            "files": []
        }
        
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                diagnosis["directories"][name]["files"] = files
                
                # Check for required FAISS files
                has_index = "index.faiss" in files
                has_pkl = "index.pkl" in files
                
                diagnosis["directories"][name]["has_required_files"] = has_index and has_pkl
                
                if not (has_index and has_pkl):
                    diagnosis["recommendations"].append(
                        f"Missing required files for {name} store. Need both index.faiss and index.pkl"
                    )
            except Exception as e:
                diagnosis["directories"][name]["error"] = str(e)
    
    return diagnosis

def test_adaptive_retrieval():
    """Test adaptive retrieval with different query types."""
    router = SmartRouter()  # Your router instance
    
    test_queries = [
        ("What are the 9 quality standards?", "Should use k=2-3"),
        ("Define safeguarding", "Should use k=3"),
        ("How do I implement quality standards in practice?", "Should use k=5-6"),
        ("Compare children's homes regulations with fostering standards", "Should use k=6-8"),
        ("Analyze the effectiveness of current child protection policies", "Should use k=8-10")
    ]
    
    for query, expectation in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {expectation}")
        
        result = router.enhanced_route_query(query)
        if result["success"]:
            print(f"Actual: k={result['total_results']}, strategy='{result['retrieval_strategy']}'")
        else:
            print(f"Failed: {result['error']}")

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Enhanced example usage with error handling
    print("🚀 Initializing Smart Router...")
    
    try:
        router = SmartRouter()
        
        # Health check first
        print("\n🔍 Health Check:")
        health = router.health_check()
        print(f"Overall Status: {health['overall_status']}")
        
        for component, status in health["components"].items():
            print(f"  {component}: {status['status']}")
        
        # Diagnostic check
        print("\n🔧 Vector Store Diagnosis:")
        diagnosis = diagnose_vector_stores()
        for store_name, store_info in diagnosis["directories"].items():
            print(f"  {store_name}: {'✅' if store_info['exists'] else '❌'} {store_info['path']}")
            if store_info.get("has_required_files"):
                print(f"    Files: ✅ Complete")
            elif store_info["exists"]:
                print(f"    Files: ❌ Incomplete - {store_info['files']}")
        
        if diagnosis["recommendations"]:
            print("\n📝 Recommendations:")
            for rec in diagnosis["recommendations"]:
                print(f"  • {rec}")
        
        # Test queries if vector stores are available
        if router.vector_stores:
            test_queries = [
                "What are the safeguarding policies for children's homes?",
                "How do I implement error handling in Python?",
                "What are the legal requirements for fostering agencies?",
                "Compare different machine learning algorithms",
                "Explain the inspection framework for children's services"
            ]
            
            print(f"\n🧪 Testing with {len(test_queries)} queries...")
            for i, query in enumerate(test_queries, 1):
                print(f"\n[{i}] Query: {query[:60]}...")
                result = router.route_query(query, k=3)
                
                if result["success"]:
                    print(f"    ✅ Provider: {result['provider']} | Results: {result['total_results']} | Time: {result['response_time']:.3f}s")
                    if result.get("used_fallback"):
                        print("    🔄 Used fallback provider")
                else:
                    print(f"    ❌ Failed: {result['error']}")
            
            # Performance summary
            print("\n" + "="*60)
            print("📊 PERFORMANCE SUMMARY")
            print("="*60)
            perf_summary = router.get_performance_summary()
            print(f"Total Queries: {perf_summary['total_queries']}")
            print(f"Available Providers: {', '.join(perf_summary['available_providers'])}")
            
            for provider, metrics in perf_summary["providers"].items():
                print(f"\n{provider.upper()}:")
                print(f"  Score: {metrics['score']:.3f}")
                print(f"  Avg Response Time: {metrics['response_time']:.3f}s")
                print(f"  Success Rate: {metrics['success_rate']:.3f}")
                print(f"  Total Queries: {metrics['total_queries']}")
        else:
            print("\n⚠️  No vector stores available - skipping query tests")
            print("   Please ensure vector databases are properly created and accessible")
        
    except Exception as e:
        logger.error(f"Failed to initialize or test SmartRouter: {e}")
        print(f"\n❌ Critical Error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if required dependencies are installed")
        print("2. Verify API keys are set correctly")
        print("3. Ensure vector databases exist and are accessible")
        print("4. Check file permissions")
    
    print("\n✨ Smart Router testing complete!")
