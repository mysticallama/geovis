#!/bin/bash
# Quick Push to Pre-Existing GitHub Repository
# Usage: ./quick_push.sh https://github.com/YOUR_USERNAME/your-repo.git

# Check if URL provided as argument
if [ -z "$1" ]; then
    echo "❌ Error: Repository URL required"
    echo ""
    echo "Usage:"
    echo "  ./quick_push.sh https://github.com/YOUR_USERNAME/your-repo.git"
    echo ""
    echo "Or run the full interactive setup:"
    echo "  ./GITHUB_SETUP.sh"
    exit 1
fi

REPO_URL="$1"

echo "🚀 Quick Push to GitHub"
echo "======================="
echo ""
echo "📡 Repository: $REPO_URL"
echo ""

# Navigate to git root
cd "/Users/colegriffiths/Documents/CG Python Projects/water_project"

# Stage all changes
echo "📦 Staging changes..."
git add -A

# Commit
echo "💾 Committing..."
git commit -m "Reorganize project structure and add rate limiting fixes

- Move all modules into planet_pipeline package directory
- Fix import statements to match new module names
- Add retry logic with exponential backoff for API rate limiting
- Reduce default parallel workers from 4 to 2 to avoid rate limits
- Add comprehensive rate limit handling in download and query modules
- Update .gitignore to exclude data directories

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Check if remote exists
existing_remote=$(git remote get-url origin 2>/dev/null)

if [[ -n "$existing_remote" ]]; then
    if [[ "$existing_remote" != "$REPO_URL" ]]; then
        echo ""
        echo "⚠️  Remote exists but points to different URL:"
        echo "   Current: $existing_remote"
        echo "   New: $REPO_URL"
        echo ""
        read -p "Update remote URL? (y/n) " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git remote set-url origin "$REPO_URL"
            echo "✅ Remote updated"
        else
            echo "❌ Cancelled"
            exit 1
        fi
    fi
else
    echo "📡 Adding remote..."
    git remote add origin "$REPO_URL"
fi

# Push
echo ""
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your code is now on GitHub!"
    echo "🔗 View at: $REPO_URL"
else
    echo ""
    echo "⚠️  Push failed. The remote may have commits you don't have."
    echo ""
    read -p "Force push? (⚠️  This will overwrite remote, y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push -u origin main --force
        echo ""
        echo "🎉 Force pushed successfully!"
        echo "🔗 View at: $REPO_URL"
    else
        echo ""
        echo "💡 To fix manually:"
        echo "   git pull origin main --rebase"
        echo "   git push -u origin main"
    fi
fi
