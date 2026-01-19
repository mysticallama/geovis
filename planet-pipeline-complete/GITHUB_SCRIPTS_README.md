# GitHub Setup Scripts

This directory includes three scripts to help you push your Planet Pipeline project to GitHub.

## 📄 Available Scripts

### 1. `quick_push.sh` - One-Liner Push ⚡

**Best for:** When you already have a GitHub repository created and just want to push code.

**Usage:**
```bash
./quick_push.sh https://github.com/YOUR_USERNAME/your-repo-name.git
```

**What it does:**
- Stages all changes
- Creates commit with detailed message
- Adds remote (or updates existing)
- Pushes to your repository
- Handles push failures with force-push option

**Example:**
```bash
./quick_push.sh https://github.com/johndoe/planet-pipeline.git
```

---

### 2. `GITHUB_SETUP.sh` - Interactive Setup Wizard 🧙

**Best for:** When you want guidance through the entire process, or aren't sure if you have a repository yet.

**Usage:**
```bash
./GITHUB_SETUP.sh
```

**What it does:**
- Stages and commits your changes
- Checks for existing remote configuration
- Asks if you have a pre-existing repository URL
- Offers to create new repository via GitHub CLI
- Handles all push scenarios with prompts
- Provides helpful instructions if things fail

**Flow:**
1. Shows what will be committed
2. Asks for confirmation
3. Commits changes
4. Detects existing remote OR asks for repository URL
5. Pushes to GitHub
6. Handles errors interactively

---

### 3. `GITHUB_INSTRUCTIONS.md` - Complete Manual Guide 📚

**Best for:** When you want to understand each step or do it manually.

**Contents:**
- Step-by-step manual instructions
- Multiple options for each step
- Troubleshooting guide
- Authentication setup
- Post-upload configuration tips

---

## 🎯 Which Script Should I Use?

### Use `quick_push.sh` if:
- ✅ You already have a GitHub repository created
- ✅ You know the repository URL
- ✅ You want the fastest option
- ✅ You're comfortable with defaults

**Example workflow:**
```bash
# Create repo on GitHub first, then:
./quick_push.sh https://github.com/username/planet-pipeline.git
```

### Use `GITHUB_SETUP.sh` if:
- ✅ You want an interactive guided experience
- ✅ You might need to create a new repository
- ✅ You're not sure what options to choose
- ✅ You want safety prompts before destructive actions

**Example workflow:**
```bash
./GITHUB_SETUP.sh
# Follow the prompts
```

### Use `GITHUB_INSTRUCTIONS.md` if:
- ✅ You want to do everything manually
- ✅ Scripts aren't working for you
- ✅ You need to understand what each command does
- ✅ You're troubleshooting an issue

---

## 🚀 Quick Start Examples

### Scenario 1: I have an existing empty GitHub repo

```bash
# Fastest way:
./quick_push.sh https://github.com/username/my-repo.git

# Or interactive:
./GITHUB_SETUP.sh
# When prompted, paste: https://github.com/username/my-repo.git
```

### Scenario 2: I need to create a new GitHub repo

```bash
# Interactive (best option):
./GITHUB_SETUP.sh
# When asked for URL, press Enter
# Follow prompts to create new repo

# Or manually:
# 1. Go to https://github.com/new
# 2. Create repo
# 3. Run: ./quick_push.sh https://github.com/username/new-repo.git
```

### Scenario 3: I already pushed before and need to update

```bash
# If remote is already configured:
cd "/Users/colegriffiths/Documents/CG Python Projects/water_project"
git add -A
git commit -m "Update pipeline code"
git push

# Or use the wizard:
./GITHUB_SETUP.sh
# It will detect existing remote and offer to push
```

---

## 🛠️ Troubleshooting

### "Permission denied"
You need to authenticate with GitHub. Run:
```bash
gh auth login  # If you have GitHub CLI
# Or set up SSH keys (see GITHUB_INSTRUCTIONS.md)
```

### "Remote already exists"
This is fine! The scripts will detect it and ask if you want to use it or update it.

### "Push rejected"
The remote has commits you don't have. Options:
1. Pull first: `git pull origin main --rebase`
2. Force push: Use the force-push option when prompted (⚠️ overwrites remote)

### Scripts won't run
Make sure they're executable:
```bash
chmod +x quick_push.sh GITHUB_SETUP.sh
```

---

## 📋 What Gets Committed?

All scripts commit the same changes:

✅ **Included:**
- All Python modules in `planet_pipeline/` package
- Documentation files (README, guides, PDF)
- Configuration templates and examples
- Setup scripts and requirements

❌ **Excluded (by .gitignore):**
- Virtual environment (`.venv/`)
- API keys and `.env` files
- Data directories (`data/`, `planet_data/`, `my_data/`)
- Python cache (`__pycache__/`)
- System files (`.DS_Store`)

---

## 💡 Tips

1. **Always review what's being committed** - Both scripts show you the git status before committing

2. **Use force push carefully** - Only force push if you're sure you want to overwrite the remote

3. **Keep your API key safe** - The `.gitignore` is configured to exclude `.env` files, but double-check!

4. **Commit messages** - All scripts use a detailed commit message explaining the changes

5. **Run from any directory** - Scripts automatically navigate to the git root

---

## 🔗 Related Files

- `.gitignore` - Controls what gets uploaded
- `START_HERE.txt` - General project setup guide
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies

---

## ❓ Questions?

If you're stuck:
1. Read [GITHUB_INSTRUCTIONS.md](GITHUB_INSTRUCTIONS.md) for detailed manual steps
2. Check the troubleshooting section above
3. Run `./GITHUB_SETUP.sh` for interactive help
4. Check git status: `git status`
5. Check remotes: `git remote -v`
