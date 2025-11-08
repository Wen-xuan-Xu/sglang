#!/usr/bin/env python3
"""
Quick test script to verify async LoRA prefetch implementation.
This script tests the basic functionality without requiring a full server setup.
"""

import sys
import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from sglang.srt.lora.mem_pool import LoRAMemoryPool, EMPTY_SLOT
        from sglang.srt.lora.lora_manager import LoRAManager, LoRAPrefetchPredictor
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_predictor():
    """Test the LoRAPrefetchPredictor class."""
    print("\nTesting LoRAPrefetchPredictor...")
    try:
        from sglang.srt.lora.lora_manager import LoRAPrefetchPredictor
        
        predictor = LoRAPrefetchPredictor(history_size=5)
        
        # Simulate some batches
        predictor.record_batch({'lora1', 'lora2'})
        predictor.record_batch({'lora2', 'lora3'})
        predictor.record_batch({'lora1', 'lora3'})
        
        # Test prediction
        predictions = predictor.predict_next_loras({'lora1'}, max_predictions=3)
        
        print(f"  Recent batches: {len(predictor.recent_batches)}")
        print(f"  Frequency map: {predictor.lora_frequency}")
        print(f"  Predictions: {predictions}")
        
        assert len(predictions) > 0, "Should have predictions"
        assert 'lora1' in predictions, "Should include current batch LoRA"
        
        print("✅ Predictor test passed")
        return True
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mem_pool_structure():
    """Test that MemoryPool has the new attributes."""
    print("\nTesting MemoryPool structure...")
    try:
        from sglang.srt.lora.mem_pool import LoRAMemoryPool
        
        # Check if the class has the new methods
        assert hasattr(LoRAMemoryPool, 'async_prefetch_lora'), "Missing async_prefetch_lora method"
        assert hasattr(LoRAMemoryPool, 'wait_for_prefetch'), "Missing wait_for_prefetch method"
        assert hasattr(LoRAMemoryPool, 'log_prefetch_metrics'), "Missing log_prefetch_metrics method"
        
        print("  ✓ async_prefetch_lora method exists")
        print("  ✓ wait_for_prefetch method exists")
        print("  ✓ log_prefetch_metrics method exists")
        
        print("✅ MemoryPool structure test passed")
        return True
    except Exception as e:
        print(f"❌ MemoryPool structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lora_manager_structure():
    """Test that LoRAManager has the new attributes."""
    print("\nTesting LoRAManager structure...")
    try:
        from sglang.srt.lora.lora_manager import LoRAManager
        
        # Check if the class has the new methods
        assert hasattr(LoRAManager, 'log_prefetch_metrics'), "Missing log_prefetch_metrics method"
        
        print("  ✓ log_prefetch_metrics method exists")
        
        print("✅ LoRAManager structure test passed")
        return True
    except Exception as e:
        print(f"❌ LoRAManager structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_stream():
    """Test basic CUDA stream functionality."""
    print("\nTesting CUDA stream availability...")
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping stream test")
            return True
        
        # Test basic CUDA stream creation
        stream = torch.cuda.Stream()
        print(f"  ✓ CUDA stream created: {stream}")
        
        # Test event creation
        event = torch.cuda.Event()
        print(f"  ✓ CUDA event created: {event}")
        
        print("✅ CUDA stream test passed")
        return True
    except Exception as e:
        print(f"❌ CUDA stream test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Async LoRA Prefetch - Quick Verification Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_predictor,
        test_mem_pool_structure,
        test_lora_manager_structure,
        test_cuda_stream,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation looks good.")
        print("\nNext steps:")
        print("1. Run full test suite: pytest test/srt/lora/ -v")
        print("2. Run performance test: pytest test/srt/test_bench_serving.py -v -s")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
