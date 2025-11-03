# GitHub Setup Instructions

## Step 1: Create a Repository on GitHub

1. Go to https://github.com/new
2. Fill in the repository details:
   - **Repository name**: `quickdraw-ml` (or your preferred name)
   - **Description**: `CNN classifier for QuickDraw dataset with appendix detection`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

3. Click "Create repository"

## Step 2: Add the Remote and Push

After creating the repository, you'll see instructions. Run these commands in your terminal:

```bash
cd /home/mcvaj/ML

# Configure git (if not done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Stage all files (respects .gitignore)
git add -A

# Create initial commit
git commit -m "Initial commit: QuickDraw ML project with dataset download and model training"

# Add GitHub remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/quickdraw-ml.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify on GitHub

Visit `https://github.com/YOUR_USERNAME/quickdraw-ml` to see your repository.

## What Gets Ignored (NOT pushed)

✓ Large data files:
- `data/raw/**/*.npy` (QuickDraw datasets)
- `data/processed/**/*.npy` (processed training data)

✓ Model files:
- `models/**/*.h5` (trained models)
- `models/**/*.pkl` (pickled objects)

✓ Python cache:
- `__pycache__/`
- `.pyc` files
- Virtual environment files

## What Gets Pushed

✓ Code files:
- `src/dataset.py`
- `src/train.py`
- `src/predict.py`
- `src/download_quickdraw_npy.py`

✓ Configuration:
- `requirements.txt`
- `.gitignore`
- `README.md`

✓ Documentation:
- Training reports
- Quick reference guides

## To Verify Before Pushing

```bash
# See what will be committed
git status

# See exactly which files are staged
git diff --cached --name-only

# Check what's being ignored
git check-ignore -v data/raw/**/*.npy
```

## Authentication Options

### Option 1: Personal Access Token (Recommended for CLI)
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token"
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token
5. When prompted for password during `git push`, paste the token

### Option 2: SSH Key
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
# Follow prompts, then add the public key to GitHub
git remote set-url origin git@github.com:YOUR_USERNAME/quickdraw-ml.git
```

### Option 3: GitHub CLI
```bash
# Install: https://cli.github.com
gh auth login
gh repo create quickdraw-ml --source=. --remote=origin --push
```
