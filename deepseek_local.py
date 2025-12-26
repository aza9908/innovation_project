import torch
from transformers import AutoTokenizer, AutoConfig
from PIL import Image
import sys
import os
import collections.abc

# 0. PATCH COLLECTIONS FOR PYTHON 3.10+
# DeepSeek-VL2 code might rely on collections.MutableMapping which moved to collections.abc
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

# 1. SETUP PATHS
# Add the local repo path explicitly to sys.path to ensure imports work
REPO_ROOT = "/home/jetson/DeepSeek-VL2"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Specific Model Snapshot Path provided by user
MODEL_PATH = "/home/jetson/DeepSeek-VL2/deepseek-vl2-tiny2/deepseek-vl2-tiny2/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa"

def load_pil_images_local(conversations):
    """
    Local implementation to load images, avoiding external dependency on deepseek_vl2 package.
    """
    pil_images = []
    for message in conversations:
        if "images" in message:
            for image_path in message["images"]:
                try:
                    pil_images.append(Image.open(image_path).convert("RGB"))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return pil_images

def get_class_from_module(module, substring):
    """Helper to find a class in a module safely, ignoring case."""
    for name in dir(module):
        if substring.lower() in name.lower() and "Config" not in name:
            return getattr(module, name)
    return None

def describe_image(image_paths, model_path=MODEL_PATH):
    """
    Runs DeepSeek-VL2 inference to describe an image.
    Includes Jetson FP16 fixes and explicit class loading.
    """
    
    # 1. Configuration for Jetson compatibility (Strict FP16)
    compute_dtype = torch.float16 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    system_prompt = "Describe this image in detail and list the main objects visible."
    
    print(f"🚀 Loading model from {model_path}...")
    print(f"🔧 Device: {device} | Dtype: {compute_dtype}")

    try:
        # 2. DYNAMICALLY IMPORT MODEL CLASS
        # We look for the modeling file in the package
        try:
            import deepseek_vl2.models.modeling_deepseek_vl_v2 as model_module
            import deepseek_vl2.models.processing_deepseek_vl_v2 as processor_module
            print("✅ Successfully imported deepseek_vl2 modules.")
            
            # Find the Model Class (e.g. DeepseekVLV2ForCausalLM vs DeepSeekVLV2ForCausalLM)
            ModelClass = get_class_from_module(model_module, "ForCausalLM")
            if ModelClass is None:
                raise ImportError(f"Could not find a 'ForCausalLM' class in {model_module}")
            print(f"✅ Found Model Class: {ModelClass.__name__}")

            # Find the Processor Class
            ProcessorClass = get_class_from_module(processor_module, "Processor")
            if ProcessorClass is None:
                print("⚠️ Could not find Processor class. Using AutoProcessor/Tokenizer fallback.")
                ProcessorClass = None
            else:
                 print(f"✅ Found Processor Class: {ProcessorClass.__name__}")

        except ImportError as e:
            print(f"❌ Critical Import Error: {e}")
            print("Ensure '/home/jetson/DeepSeek-VL2/deepseek_vl2' exists and contains __init__.py")
            return

        # 3. Load Processor/Tokenizer
        if ProcessorClass:
            vl_chat_processor = ProcessorClass.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = vl_chat_processor.tokenizer
        else:
            from transformers import AutoProcessor
            try:
                vl_chat_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = vl_chat_processor.tokenizer
            except:
                print("⚠️ AutoProcessor failed. Loading Tokenizer only.")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                vl_chat_processor = tokenizer # Fallback, might crash on pixel_values

        # 4. LOAD MODEL
        print("⏳ Loading model weights (this may take a moment)...")
        # device_map="auto" is critical for Jetson to offload layers to CPU if GPU RAM is full.
        # torch_dtype=compute_dtype ensures layers load in FP16.
        model = ModelClass.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            device_map="auto" 
        )
        
        # REMOVED: model = model.to(compute_dtype)
        # Reason: When using device_map="auto", Accelerate handles the device/dtype. 
        # Manually casting the whole model afterwards triggers "RuntimeError: You can't move a model..."

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Prepare Inputs
    conversation = [
        {
            "role": "User",
            "content": f"<image>\n{system_prompt}",
            "images": image_paths if isinstance(image_paths, list) else [image_paths]
        },
        {"role": "Assistant", "content": ""}
    ]

    pil_images = load_pil_images_local(conversation)
    
    try:
        # Prepare inputs
        # If we have a real processor, use it. If we only have a tokenizer, this part needs care.
        if hasattr(vl_chat_processor, 'image_processor') or hasattr(vl_chat_processor, 'process'):
             prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(device)
        else:
            # Emergency manual fallback if processor failed entirely
            print("⚠️ Using manual input preparation (Processor failed)...")
            inputs = tokenizer(
                conversation[0]["content"], # This is hacky, assumes text only
                return_tensors="pt"
            )
            prepare_inputs = inputs.to(device)
            # This path will likely fail for VL models because image tokens aren't handled.
            # But the 'if ProcessorClass' block above should prevent this.

        # 6. CRITICAL FIX: Cast Image Embeddings to FP16
        # This fixes the original error "Input type (BFloat16) and bias type (Half) should be the same"
        if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=compute_dtype)

        # Run Generation
        print("✨ Analyzing image...")
        
        with torch.no_grad():
            outputs = model.generate(
                prepare_inputs.input_ids,
                images=getattr(prepare_inputs, 'pixel_values', None),
                images_seq_mask=getattr(prepare_inputs, 'images_seq_mask', None),
                images_spatial_crop=getattr(prepare_inputs, 'images_spatial_crop', None),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512, 
                do_sample=False,
                use_cache=True
            )

        # Decode Output
        description = tokenizer.decode(outputs[0][prepare_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print("\n📝 Image Description:")
        print("-" * 40)
        print(description)
        print("-" * 40)
        
        return description

    except Exception as e:
        print(f"\n❌ Error during inference prep: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check for command line argument first, otherwise use default test image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "/home/jetson/DeepSeek-VL2/test.jpg"
        print(f"No argument provided. Using default image: {img_path}")

    if os.path.exists(img_path):
        describe_image([img_path])
    else:
        print(f"File not found: {img_path}")
        print("Usage: python deepseek_local_fix.py <path_to_image>")
