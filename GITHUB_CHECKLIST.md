# GitHub Repository Setup - Complete Checklist

## ✅ What's Been Done

### 1. Git Repository Initialized
- ✅ Local repository created: `/home/mcvaj/ML/.git`
- ✅ Git configured with user details

### 2. .gitignore Updated
Large files that will NOT be pushed to GitHub:
- ✅ `data/raw/**/*.npy` - Downloaded QuickDraw datasets (~2.1GB)
- ✅ `data/raw/**/*.ndjson` - Raw ndjson files
- ✅ `data/processed/**/*.npy` - Processed training data
- ✅ `data/processed/**/*.pkl` - Pickle files
- ✅ `models/**/*.h5` - Trained models (Keras)
- ✅ `models/**/*.pkl` - Model pickles
- ✅ `__pycache__/`, `*.pyc` - Python cache
- ✅ Virtual environment files

### 3. Files Ready to Push
Code and documentation files that WILL be pushed:
- ✅ `src/dataset.py` - Dataset loading and preprocessing
- ✅ `src/train.py` - Model training script
- ✅ `src/predict.py` - Inference script
- ✅ `src/download_quickdraw_npy.py` - Download script (new)
- ✅ `src/download_quickdraw.py` - Original download script
- ✅ `src/generate_negatives.py` - Negative sample generation
- ✅ `src/appendix_loader.py` - Custom loader
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Project documentation
- ✅ `TRAINING_REPORT.md` - Training results
- ✅ `.gitignore` - Ignore rules
- ✅ `models/training_history.png` - Training plots
- ✅ `models/confusion_matrix.png` - Evaluation plots
- ✅ All documentation files

## 📋 Next Steps - Create Repository on GitHub

### Option 1: Use GitHub Web Interface (Easiest)

1. **Open GitHub**: Go to https://github.com/new

2. **Fill in details**:
   - Repository name: `quickdraw-ml`
   - Description: `CNN classifier for QuickDraw dataset with appendix detection`
   - Visibility: Public (recommended for portfolio) or Private
   - **IMPORTANT**: Do NOT initialize with README/gitignore/license

3. **Create repository** and copy the repository URL

### Option 2: Use GitHub CLI

```bash
gh auth login              # Authenticate with GitHub
gh repo create quickdraw-ml --public --source=.
```

## 🚀 Push Your Code to GitHub

After creating the repository on GitHub, run these commands:

```bash
cd /home/mcvaj/ML

# Configure git user (first time only)
git config user.name "Your Full Name"
git config user.email "your.email@gmail.com"

# Stage all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: QuickDraw ML project with CNN classifier

- Implemented binary classification model for QuickDraw detection
- Dataset download script for 21 categories (~2.1GB)
- Training pipeline with validation and early stopping
- Model evaluation with AUC metrics
- Support for inference on new drawings"

# Add GitHub remote (replace URL with your repository)
git remote add origin https://github.com/YOUR_USERNAME/quickdraw-ml.git

# Rename branch to main (recommended)
git branch -M main

# Push to GitHub
git push -u origin main
```

## 🔑 Authentication

When prompted for a password during `git push`:

### Using Personal Access Token (Recommended)

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Click "Tokens (classic)" → "Generate new token (classic)"
3. Select scope: `repo` (full control of repositories)
4. Copy the token
5. Paste as password when prompted

### Or Use SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@gmail.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Update remote URL
git remote set-url origin git@github.com:YOUR_USERNAME/quickdraw-ml.git
```

## 📊 Verify Before Pushing

```bash
# Check status
git status

# See what will be pushed
git log --oneline

# Verify large files are ignored
git check-ignore -v data/raw/*.npy data/processed/*.npy models/*.h5

# Estimate repository size
du -sh .git
```

## 📝 After Pushing

1. **Visit your repository**: `https://github.com/YOUR_USERNAME/quickdraw-ml`
2. **Add README details** if needed (GitHub will show it automatically)
3. **Add topics** for discoverability: `machine-learning`, `quickdraw`, `tensorflow`, `cnn`
4. **Future updates**: Use `git add`, `git commit`, `git push`

## ⚠️ Important Notes

- **Large files are excluded**: Data and models won't be pushed (2.1GB+ savings!)
- **Reproducibility**: Users can run `download_quickdraw_npy.py` to get the data
- **Models**: Include model weights separately or document how to train
- **Private data**: If you have sensitive data, keep repo private

## 📚 Quick Git Commands Reference

```bash
# Check status
git status

# Stage specific file
git add src/train.py

# Stage everything (respects .gitignore)
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# See commit history
git log --oneline

# See what changed
git diff

# Create new branch
git checkout -b feature-name

# Switch branch
git checkout main
```

---

**Once completed, share your repository link!** 🎉
