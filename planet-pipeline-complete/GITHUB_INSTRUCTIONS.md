# How to Push Planet Pipeline to GitHub

## 🚀 Super Quick Start

**If you have a pre-existing GitHub repository:**

```bash
# Option 1: One-liner (fastest!)
./quick_push.sh https://github.com/YOUR_USERNAME/your-repo-name.git

# Option 2: Interactive
./GITHUB_SETUP.sh
# When prompted, paste your repository URL
```

**If you need to create a new repository:**

```bash
./GITHUB_SETUP.sh
# Press Enter when asked for URL, follow prompts to create new repo
```

---

## Quick Start (Automated)

Simply run the automated setup script:

```bash
./GITHUB_SETUP.sh
```

The script will:
1. Stage and commit all your changes
2. Detect if you already have a remote configured
3. Ask if you have a pre-existing repository URL
4. Push to your repository (existing or new)

### Using with Pre-Existing Repository

When prompted, simply paste your repository URL:

```
Enter repository URL: https://github.com/YOUR_USERNAME/your-existing-repo.git
```

The script handles everything automatically!

---

## Manual Setup (Step-by-Step)

### Step 0: (Optional) Check for Existing Remote

```bash
# Check if you already have a remote configured
git remote -v
```

If you see a remote listed, you can skip to Step 4 and just push to it.

### Step 1: Navigate to Git Root

```bash
cd "/Users/colegriffiths/Documents/CG Python Projects/water_project"
```

### Step 2: Stage All Changes

```bash
git add -A
```

This stages:
- All new files in `planet-pipeline-complete/`
- Deletion of old files in parent directory
- Updated `.gitignore`

### Step 3: Check What Will Be Committed

```bash
git status
```

You should see:
- ✅ New files in `planet-pipeline-complete/`
- ✅ Deleted old module files
- ❌ `.DS_Store` files (ignored by .gitignore)

### Step 4: Commit Changes

```bash
git commit -m "Reorganize project structure and add rate limiting fixes

- Move all modules into planet_pipeline package directory
- Fix import statements to match new module names
- Add retry logic with exponential backoff for API rate limiting
- Reduce default parallel workers from 4 to 2 to avoid rate limits
- Add comprehensive rate limit handling in download and query modules
- Update .gitignore to exclude data directories

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Step 5: Push to GitHub Repository

#### Option A: Using Pre-Existing Repository (Recommended if you already have one)

If you already have a GitHub repository created, simply add it as the remote:

```bash
# Add your existing repository URL
git remote add origin https://github.com/YOUR_USERNAME/your-repo-name.git

# Push to the repository
git branch -M main
git push -u origin main
```

**Note:** Replace `YOUR_USERNAME/your-repo-name` with your actual repository path.

**If the push fails** (remote has commits you don't have):

```bash
# Option 1: Pull and merge first (safer)
git pull origin main --rebase
git push -u origin main

# Option 2: Force push (⚠️ overwrites remote)
git push -u origin main --force
```

#### Option B: Using GitHub CLI (Recommended for new repositories)

If you have GitHub CLI installed:

```bash
# Create public repository
gh repo create planet-pipeline --public --source=. --remote=origin --push

# OR create private repository
gh repo create planet-pipeline --private --source=. --remote=origin --push
```

This will automatically:
- Create the repository on GitHub
- Add remote origin
- Push your code

#### Option C: Manual Creation (via Web Browser)

1. Go to [https://github.com/new](https://github.com/new)

2. Fill in repository details:
   - **Repository name:** `planet-pipeline`
   - **Description:** `Production-ready Planet Labs satellite imagery processing pipeline for ML workflows`
   - **Visibility:** Choose Public or Private
   - **❌ DO NOT** check "Add a README file"
   - **❌ DO NOT** check "Add .gitignore"
   - **❌ DO NOT** choose a license (you can add later)

3. Click **"Create repository"**

4. On the next page, copy the repository URL (looks like: `https://github.com/YOUR_USERNAME/planet-pipeline.git`)

5. Back in your terminal, run:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/planet-pipeline.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 6: Verify Upload

Visit your repository on GitHub:
```
https://github.com/YOUR_USERNAME/planet-pipeline
```

You should see:
- ✅ All files in `planet-pipeline-complete/` directory
- ✅ README.md displayed on the main page
- ✅ Python files with syntax highlighting

---

## Troubleshooting

### "remote origin already exists"

If you get this error, the remote is already configured. Check it:

```bash
git remote -v
```

To update the remote URL:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/planet-pipeline.git
```

### "failed to push some refs"

This usually means the remote has changes you don't have locally. To force push (⚠️ use with caution):

```bash
git push -u origin main --force
```

Or pull first, then push:

```bash
git pull origin main --rebase
git push -u origin main
```

### "permission denied"

You need to authenticate with GitHub. Options:

1. **GitHub CLI** (easiest):
   ```bash
   gh auth login
   ```

2. **Personal Access Token:**
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope
   - Use token as password when pushing

3. **SSH Key:**
   - Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
   - Add to GitHub: Settings → SSH and GPG keys
   - Change remote to SSH: `git remote set-url origin git@github.com:YOUR_USERNAME/planet-pipeline.git`

---

## Recommended Repository Settings

After creating the repository, configure these settings on GitHub:

### 1. Add Topics (helps others discover your project)

Go to repository → About (gear icon) → Topics:
- `planet-labs`
- `satellite-imagery`
- `remote-sensing`
- `machine-learning`
- `python`
- `geospatial`
- `earth-observation`

### 2. Add Description

```
Production-ready Python pipeline for querying, downloading, and processing Planet Labs satellite imagery for ML workflows. Supports spectral indices, preprocessing, and PyTorch/TensorFlow data preparation.
```

### 3. Enable Issues and Discussions

Settings → Features:
- ✅ Issues
- ✅ Discussions (optional, for community Q&A)

### 4. Add License (Optional)

Common choices:
- **MIT License**: Most permissive, allows commercial use
- **Apache 2.0**: Similar to MIT, includes patent grant
- **GPL-3.0**: Copyleft, requires derivative works to be open source

Add via: Add file → Create new file → Name: `LICENSE`

### 5. Branch Protection (Optional)

For collaboration:
- Settings → Branches → Add rule
- Branch name pattern: `main`
- ✅ Require pull request reviews before merging

---

## Next Steps After Pushing

### 1. Add GitHub Actions (Optional)

Create `.github/workflows/python-tests.yml` for automated testing:

```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest
```

### 2. Add Badges to README

At the top of README.md:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/planet-pipeline)](https://github.com/YOUR_USERNAME/planet-pipeline/issues)
```

### 3. Create Release

When ready to version your code:

```bash
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

Then create release on GitHub: Releases → Draft a new release

---

## Questions?

- Check git status: `git status`
- View commit history: `git log --oneline`
- View remotes: `git remote -v`
- Get help: `git help <command>`
