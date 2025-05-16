# 🎨 AnimeGAN Face Stylizer

A Streamlit web application that converts human face images into anime-style portraits using the AnimeGAN2/AnimeGAN3 model.

---

## 📌 About the Project

This project uses a pre-trained AnimeGAN2 model to stylize human faces into anime-style artwork and AnimeGAN3 for landscape images. It provides an intuitive interface for users to upload an image, select an output scale, view both original and stylized versions, and download the result.

---

## ✨ Highlights

- ✅ **AnimeGAN Model**: Powered by a pre-trained AnimeGAN model from Torch Hub.
- ⚡ **Optimized Performance**: Uses GPU acceleration when available.
- 🔍 **Scale Options**: Supports multiple output size scales (Original, 1.5x, 2x, 3x).
- 🖼️ **Side-by-Side Display**: Displays original and stylized images next to each other.
- 📥 **Downloadable Output**: Enables easy downloading of the output image in PNG format.

---

## ⚠️ Limitations

- 🎭 AnimeGAN2 Works best with clear, front-facing human faces and AnimeGAN3 works best for landscape images
- 🌐 Requires internet access to load the model from Torch Hub (GitHub) on the first run.
- 📁 Accepts only `.jpg`, `.jpeg`, and `.png` image formats.
- 🔍 No face detection validation – ensure the image includes a face.

---

## 🛠️ How to Run

### 1. Clone the Repository (if applicable)
```bash
git clone https://github.com/vignesh2396/ICBP-2.0-Project.git
cd ICBP-2.0-Project
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed, then run:
```bash
pip install streamlit torch torchvision pillow
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 🚀 Using the App

1. Upload an image (`.jpg`, `.jpeg`, or `.png`).
2. Choose the output scale (Original, 1.5x, 2x, or 3x).
3. Wait for the anime-style version to be generated.
4. View the original and stylized image side-by-side.
5. Click the **Download Output Image** button to save your anime portrait.

---

## 🧯 Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| `ModuleNotFoundError` | Missing library | Install required packages using `pip install` |
| `torch.hub.load` fails | Internet/GitHub access issue | Ensure internet is available and GitHub is reachable |
| App crashes on image upload | Corrupt or unsupported image | Use clean RGB `.jpg`/`.png` images with faces |
| CUDA not detected | GPU unavailable | App will use CPU automatically; no action needed |
| Freezes with large images | High memory usage | Try reducing image resolution |

---

## 📄 License

This project is for educational and research purposes. Refer to AnimeGAN2’s original repository for licensing terms.

---

## 🙋‍♂️ Questions or Contributions?

Feel free to raise issues or pull requests if you'd like to contribute or need support.
