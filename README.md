# DeepSeek-VL2 Local Implementation for NVIDIA Jetson

A lightweight implementation of DeepSeek-VL2 vision-language model optimized for NVIDIA Jetson devices with FP16 support and camera integration.

## 📋 Project Overview

This project enables running DeepSeek-VL2-Tiny vision-language model locally on NVIDIA Jetson devices (Nano, Xavier NX, AGX Orin) for real-time image understanding and description tasks.

**Key Features:**
- ✅ Optimized for Jetson FP16 inference
- ✅ USB/CSI camera integration
- ✅ Automatic dtype handling
- ✅ Memory-efficient with device_map="auto"
- ✅ Fixed collections.MutableMapping compatibility for Python 3.10+

## 🛠️ Hardware Requirements

- **NVIDIA Jetson Device** (Nano, Xavier NX, Orin Nano, AGX Orin)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 10GB+ free space for model weights
- **USB Camera** (or CSI camera module)

## 📦 Software Requirements

### System Setup
```bash
# JetPack 5.0+ (includes CUDA, cuDNN)
# Python 3.8+
# PyTorch with CUDA support
```

### Python Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow opencv-python
```

## 🚀 Installation

### Step 1: Clone DeepSeek-VL2 Repository
```bash
cd /home/jetson
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
cd DeepSeek-VL2
```

### Step 2: Download Model Weights
Download the `deepseek-vl2-tiny` model from Hugging Face:

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login (optional, for private models)
huggingface-cli login

# Download model
huggingface-cli download deepseek-ai/deepseek-vl2-tiny \
  --local-dir /home/jetson/DeepSeek-VL2/deepseek-vl2-tiny
```

**OR** manually download from: https://huggingface.co/deepseek-ai/deepseek-vl2-tiny

### Step 3: Install DeepSeek-VL2 Package
```bash
cd /home/jetson/DeepSeek-VL2
pip install -e .
```

### Step 4: Clone This Project
```bash
cd /home/jetson
git clone https://github.com/aza9908/innovation_project.git
cd innovation_project
```

### Step 5: Update Model Path
Edit `deepseek_local.py` and update the `MODEL_PATH` variable to match your snapshot directory:

```python
MODEL_PATH = "/home/jetson/DeepSeek-VL2/deepseek-vl2-tiny/models--deepseek-ai--deepseek-vl2-tiny/snapshots/YOUR_SNAPSHOT_ID"
```

To find your snapshot ID:
```bash
ls /home/jetson/DeepSeek-VL2/deepseek-vl2-tiny/models--deepseek-ai--deepseek-vl2-tiny/snapshots/
```

## 💻 Usage

### Test Camera
First, verify your camera works:

```bash
python test_cam.py
```

This will:
- Capture one frame from camera
- Save it as `test.jpg`
- Exit automatically

### Run Image Analysis
Analyze any image:

```bash
# Using default test image
python deepseek_local.py

# Using custom image path
python deepseek_local.py /path/to/your/image.jpg

# Using camera-captured image
python deepseek_local.py test.jpg
```

### Example Workflow
```bash
# 1. Capture image from camera
python test_cam.py

# 2. Analyze the captured image
python deepseek_local.py test.jpg
```

## 📂 Project Structure

```
innovation_project/
├── README.md              # This file
├── deepseek_local.py      # Main inference script
├── test_cam.py            # Camera test utility
└── requirements.txt       # Python dependencies
```

## 🔧 Technical Details

### Key Fixes for Jetson

1. **FP16 Dtype Compatibility**
```python
compute_dtype = torch.float16
prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=compute_dtype)
```

2. **Device Map for Memory Management**
```python
model = ModelClass.from_pretrained(
    model_path,
    torch_dtype=compute_dtype,
    device_map="auto"  # Automatic layer offloading
)
```

3. **Collections.MutableMapping Patch**
```python
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
```

4. **Dynamic Class Loading**
```python
ModelClass = get_class_from_module(model_module, "ForCausalLM")
ProcessorClass = get_class_from_module(processor_module, "Processor")
```

## 🐛 Troubleshooting

### Issue: "RuntimeError: Input type and bias type should be the same"
**Solution:** Already fixed in `deepseek_local.py` with pixel_values casting to FP16

### Issue: "CUDA out of memory"
**Solution:**
- Use smaller batch size
- Close other GPU-intensive applications
- Consider using swap memory:
```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: "Camera not opened"
**Solution:**
- Check camera connection: `ls /dev/video*`
- Try different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- For CSI camera, use GStreamer pipeline instead

### Issue: "Module not found: deepseek_vl2"
**Solution:**
```bash
cd /home/jetson/DeepSeek-VL2
pip install -e .
```

### Issue: Model path not found
**Solution:** Verify your model path:
```bash
find /home/jetson/DeepSeek-VL2 -name "config.json"
```

## 📊 Performance Benchmarks

| Device | Model | Inference Time | Memory Usage |
|--------|-------|----------------|--------------|
| Jetson Orin Nano 8GB | deepseek-vl2-tiny | ~3-5s | ~2.5GB |
| Jetson Xavier NX | deepseek-vl2-tiny | ~5-7s | ~2.8GB |
| Jetson Nano 4GB | deepseek-vl2-tiny | ~15-20s | ~3.5GB* |

*Requires swap memory on Jetson Nano

## 🔗 Related Links

- [DeepSeek-VL2 Official Repo](https://github.com/deepseek-ai/DeepSeek-VL2)
- [DeepSeek-VL2-Tiny Model](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson)

## 📝 Example Output

```
🚀 Loading model from /home/jetson/DeepSeek-VL2/deepseek-vl2-tiny...
🔧 Device: cuda | Dtype: torch.float16
✅ Successfully imported deepseek_vl2 modules.
✅ Found Model Class: DeepseekVLV2ForCausalLM
✅ Found Processor Class: DeepseekVLV2Processor
⏳ Loading model weights (this may take a moment)...
✨ Analyzing image...

📸 Image Description:
----------------------------------------
The image shows a wooden desk with a laptop, 
a coffee mug, a notebook, and a pen. The laptop 
screen displays code in a text editor. Natural 
light comes from a window on the left side.
----------------------------------------
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project follows the license of DeepSeek-VL2. See their repository for details.

## 👤 Author

**Azamat**
- GitHub: [@aza9908](https://github.com/aza9908)

## 🙏 Acknowledgments

- DeepSeek AI team for the VL2 model
- NVIDIA for Jetson platform
- Hugging Face for model hosting

---

**Last Updated:** December 26, 2024
