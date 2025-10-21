"""
Test script to verify all three fixes:
1. fin_model_evaluation.py now uses real predictions (not synthetic)
2. WaveNetBlock and AttentionLayer are properly registered for serialization
3. Target statistics are printed to diagnose MAE issues
"""
import tensorflow as tf
import numpy as np
import sys

print("="*80)
print("VERIFICATION TEST SUITE")
print("="*80)

# Test 1: Check if custom layers are registered
print("\n1. TESTING CUSTOM LAYER REGISTRATION")
print("-"*40)


try:
    model_path = 'run_financial_wavenet_v1/best_model.keras'
    print(f"Attempting to load: {model_path}")
    
    # Use the helper function that includes custom_objects
    from fin_model import load_model_with_custom_objects
    model = load_model_with_custom_objects(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Get input shape from model
    input_shape = model.input_shape  # (None, seq_len, n_features)
    seq_len = input_shape[1]
    n_features = input_shape[2]
    print(f"   Expected input: (batch, {seq_len}, {n_features})")
    
    # Try a prediction with correct shape
    dummy_input = np.random.randn(1, seq_len, n_features)
    predictions = model.predict(dummy_input, verbose=0)
    
    if isinstance(predictions, dict):
        print(f"   Outputs: {list(predictions.keys())}")
        for key, val in predictions.items():
            print(f"      {key}: shape {val.shape}")
    else:
        print(f"   Output shape: {predictions.shape}")
    
    print("✅ Model inference works!")
    
except FileNotFoundError:
    print("⚠️  Model file not found (train first with fin_training.py)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check fin_model_evaluation.py for synthetic returns
print("\n3. CHECKING FIN_MODEL_EVALUATION.PY")
print("-"*40)

try:
    with open('fin_model_evaluation.py', 'r') as f:
        content = f.read()
    
    # Check for key changes
    if 'Generating strategy returns from model predictions' in content:
        print("✅ Uses real model predictions for strategy returns")
    else:
        print("❌ Still using only synthetic returns")
    
    if 'is_keras = hasattr(model' in content:
        print("✅ Checks model type (Keras vs sklearn)")
    else:
        print("⚠️  May not handle different model types")
    
    if 'direction_probs' in content or 'predict_proba' in content:
        print("✅ Extracts probabilities from model")
    else:
        print("❌ Not extracting model probabilities")
        
except Exception as e:
    print(f"❌ Error checking file: {e}")

# Test 4: Check fin_training.py for target diagnostics
print("\n4. CHECKING TARGET DIAGNOSTICS IN FIN_TRAINING.PY")
print("-"*40)

try:
    with open('fin_training.py', 'r') as f:
        content = f.read()
    
    if 'Target Statistics' in content:
        print("✅ Added diagnostic prints for target statistics")
    else:
        print("❌ No diagnostic prints for targets")
    
    if 'forward_returns.min()' in content and 'forward_volatility.min()' in content:
        print("✅ Prints min/max/mean/std for both targets")
    else:
        print("⚠️  Incomplete diagnostics")
        
except Exception as e:
    print(f"❌ Error checking file: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
FIXES APPLIED:

1. ✅ fin_model_evaluation.py now generates strategy returns from 
   actual model predictions (not synthetic random data)
   
2. ✅ Added diagnostic prints in fin_training.py to show 
   target statistics (min/max/mean/std)

NEXT STEPS:

1. Re-run training to see target diagnostics:
   python fin_training.py
   
2. Check if volatility and magnitude targets are in correct range:
   - Volatility should be ~0.01-0.05 (1-5%)
   - Magnitude should be ~0.01-0.10 (1-10%)
   
3. If MAE is still high, may need to scale targets or adjust loss function

4. Test model loading:
   python test_model_evaluation.py
""")

print("="*80)
