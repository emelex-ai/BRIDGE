#!/usr/bin/env python3
"""
Runner script for sliding window attention test.

This script runs the sliding window attention comparison test on GPU.
"""

import os
import sys

# Add the bridge model directory to Python path
bridge_model_path = os.path.join(os.path.dirname(__file__), "bridge", "domain", "model")
sys.path.insert(0, bridge_model_path)

# Import and run the test
try:
    from test_local_attention_simplified import test_sliding_window_attention_gpu

    print("Starting Simplified Sliding Window Attention Test")
    print("=" * 60)

    success = test_sliding_window_attention_gpu()

    if success:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure you're in the correct directory and have the required dependencies."
    )
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
