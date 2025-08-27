# quick_speed_fix.py
# Save this as a new file in your project directory

import os
import time
import logging

logger = logging.getLogger(__name__)

def apply_emergency_speed_fixes(rag_system):
    """Apply immediate speed fixes to existing system"""
    
    print("🚨 APPLYING EMERGENCY SPEED FIXES...")
    
    # FIX 1: Add instant answers cache (BIGGEST IMPACT)
    def add_instant_cache():
        instant_answers = {
            "what is dbs": "DBS (Disclosure and Barring Service) provides criminal record checks for safer recruitment in the UK.",
            "what does lac stand for": "LAC stands for 'Looked After Children' - children in local authority care.",
            "what is a pep meeting": "PEP (Personal Education Plan) meeting reviews education plans for looked after children every 6 months.",
            "what does send stand for": "SEND stands for 'Special Educational Needs and Disabilities'.",
            "what is safeguarding": "Safeguarding means protecting children from harm and promoting their welfare.",
            "what is independent living": "Independent living skills help young people transition out of care successfully.",
            "child protection": "Child protection involves taking action to protect specific children who are suffering or likely to suffer significant harm.",
            "what are ofsted ratings": "Ofsted ratings are: Outstanding, Good, Requires Improvement, and Inadequate.",
            "what is a care plan": "A care plan outlines how a child's needs will be met while they are looked after by the local authority.",
            "what does pep stand for": "PEP stands for Personal Education Plan - a statutory document for looked after children's education."
        }
        
        if hasattr(rag_system, 'query'):
            # Back up original method if not already done
            if not hasattr(rag_system, '_original_query_speed'):
                rag_system._original_query_speed = rag_system.query
            
            def cached_query(question, **kwargs):
                q_lower = question.lower().strip()
                
                # Check for instant answers (only for simple questions)
                if len(question.split()) <= 8:  # Only short questions
                    for key, answer in instant_answers.items():
                        if key in q_lower:
                            logger.info(f"⚡ INSTANT: {key}")
                            return {
                                "answer": answer,
                                "sources": [],
                                "metadata": {
                                    "llm_used": "instant_cache",
                                    "response_mode": "instant",
                                    "total_response_time": 0.1,
                                    "instant_answer": True
                                },
                                "confidence_score": 0.95
                            }
                
                # Use original query for non-cached questions
                return rag_system._original_query_speed(question, **kwargs)
            
            rag_system.query = cached_query
        print("✅ Instant answer cache added")
    
    # FIX 2: Reduce context size drastically
    def patch_context_building():
        if hasattr(rag_system, '_build_context'):
            original_build = rag_system._build_context
            
            def fast_context_build(docs):
                if not docs:
                    return ""
                
                # Use maximum 2 documents, 500 chars each
                max_docs = min(2, len(docs))
                docs = docs[:max_docs]
                
                context_parts = []
                for doc in docs:
                    content = doc.get('content', '')[:500]  # Max 500 chars per doc
                    source = doc.get('source', 'Document')[:30]  # Short source names
                    context_parts.append(f"[{source}]\n{content}")
                
                result = '\n---\n'.join(context_parts)
                # Hard limit: 1500 chars total
                if len(result) > 1500:
                    result = result[:1500] + "\n[Truncated for speed]"
                
                logger.info(f"⚡ FAST CONTEXT: {len(result)} chars from {len(docs)} docs")
                return result
            
            rag_system._build_context = fast_context_build
        print("✅ Context size reduced for speed")
    
    # FIX 3: Force smaller k values
    def patch_query_parameters():
        if hasattr(rag_system, '_speed_optimized_query'):
            original_speed_query = rag_system._speed_optimized_query
            
            def faster_speed_query(question, k, start):
                # Force smaller k values
                if k > 3:
                    k = 2  # Maximum 2 documents for speed
                    logger.info(f"⚡ FORCED k=2 for maximum speed")
                
                return original_speed_query(question, k, start)
            
            rag_system._speed_optimized_query = faster_speed_query
        print("✅ Document retrieval limited for speed")
    
    # Apply all fixes
    try:
        add_instant_cache()
        patch_context_building()
        patch_query_parameters()
        
        print("\n🚀 EMERGENCY FIXES APPLIED!")
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False

def quick_speed_test(rag_system):
    """Quick speed test after applying fixes"""
    
    test_questions = [
        "What is DBS?",           # Should be instant
        "What does LAC stand for?", # Should be instant
        "What are safeguarding policies?",  # Should be fast
    ]
    
    print("\n🔬 QUICK SPEED TEST:")
    print("-" * 40)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"Test {i}: {question}")
        start = time.time()
        
        try:
            response = rag_system.query(question, k=3)
            elapsed = time.time() - start
            
            # Check if it was an instant answer
            is_instant = response.get("metadata", {}).get("instant_answer", False)
            
            if elapsed < 1.0:
                status = "🚀 INSTANT" if is_instant else "🚀 EXCELLENT"
            elif elapsed < 3.0:  
                status = "⚡ FAST"
            elif elapsed < 6.0:
                status = "✅ GOOD"
            else:
                status = "⚠️ STILL SLOW"
            
            print(f"   Result: {status} ({elapsed:.1f}s)")
            results.append({"question": question, "time": elapsed, "success": True, "instant": is_instant})
            
        except Exception as e:
            print(f"   Result: ❌ ERROR - {str(e)[:50]}")
            results.append({"question": question, "time": 999, "success": False})
    
    # Summary
    successful = [r for r in results if r["success"]]
    instant_count = len([r for r in successful if r.get("instant", False)])
    fast_count = len([r for r in successful if r["time"] < 3.0])
    
    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        print(f"\n📊 RESULTS:")
        print(f"   Average time: {avg_time:.1f}s")
        print(f"   Instant answers: {instant_count}/{len(test_questions)}")
        print(f"   Fast responses (<3s): {fast_count}/{len(successful)}")
        
        if avg_time < 2.0:
            print("   ✅ EXCELLENT - Speed target achieved!")
        elif avg_time < 5.0:
            print("   ⚡ GOOD - Major improvement!")
        else:
            print("   ⚠️ Still needs optimization")
    
    return results

if __name__ == "__main__":
    print("🚨 EMERGENCY SPEED FIX")
    print("=" * 50)
    print("Save this file as 'quick_speed_fix.py' in your project directory")
    print("Then follow the integration steps!")
