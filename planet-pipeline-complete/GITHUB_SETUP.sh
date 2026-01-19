#!/bin/bash
# GitHub Repository Setup Script for Planet Pipeline
# Run this script to push your project to GitHub

echo "🚀 Planet Pipeline - GitHub Setup"
echo "================================="
echo ""

# Navigate to git root
cd "/Users/colegriffiths/Documents/CG Python Projects/water_project"

echo "📍 Current directory: $(pwd)"
echo ""

# Stage all changes
echo "📦 Staging all changes..."
git add -A

# Show status
echo ""
echo "📊 Git Status:"
git status --short

# Ask for confirmation
echo ""
read -p "❓ Do you want to commit these changes? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled. No changes committed."
    exit 1
fi

# Commit changes
echo ""
echo "💾 Committing changes..."
git commit -m "Reorganize project structure and add rate limiting fixes

- Move all modules into planet_pipeline package directory
- Fix import statements to match new module names
- Add retry logic with exponential backoff for API rate limiting
- Reduce default parallel workers from 4 to 2 to avoid rate limits
- Add comprehensive rate limit handling in download and query modules
- Update .gitignore to exclude data directories

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

echo ""
echo "✅ Changes committed successfully!"
echo ""

# Check if remote already exists
existing_remote=$(git remote get-url origin 2>/dev/null)

if [[ -n "$existing_remote" ]]; then
    echo "📡 Existing remote found: $existing_remote"
    echo ""
    read -p "❓ Push to existing remote? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📤 Pushing to existing remote..."
        git push -u origin main

        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 Successfully pushed to: $existing_remote"
        else
            echo ""
            echo "⚠️  Push failed. You may need to pull first or force push."
            echo ""
            read -p "❓ Force push? (⚠️  This will overwrite remote, y/n) " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git push -u origin main --force
                echo ""
                echo "🎉 Force pushed to: $existing_remote"
            fi
        fi
        exit 0
    else
        echo ""
        read -p "❓ Update remote URL instead? (y/n) " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            read -p "Enter new repository URL: " new_url

            if [[ -n "$new_url" ]]; then
                git remote set-url origin "$new_url"
                echo "✅ Remote updated to: $new_url"
                echo ""
                read -p "❓ Push now? (y/n) " -n 1 -r
                echo ""

                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    git push -u origin main
                    echo ""
                    echo "🎉 Successfully pushed to: $new_url"
                fi
                exit 0
            fi
        fi
    fi
fi

# No existing remote - ask if user has a pre-existing repo
echo "🤔 No remote configured yet."
echo ""
echo "Do you have a pre-existing GitHub repository URL?"
echo ""
read -p "❓ Enter repository URL (or press Enter to create new): " repo_url

if [[ -n "$repo_url" ]]; then
    # User provided a URL - use existing repository
    echo ""
    echo "📡 Adding remote: $repo_url"
    git remote add origin "$repo_url"

    echo ""
    read -p "❓ Push to this repository now? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📤 Pushing to repository..."
        git branch -M main
        git push -u origin main

        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 Successfully pushed to: $repo_url"
        else
            echo ""
            echo "⚠️  Push failed. The remote may have commits you don't have."
            echo ""
            read -p "❓ Force push? (⚠️  This will overwrite remote, y/n) " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git push -u origin main --force
                echo ""
                echo "🎉 Force pushed to: $repo_url"
            else
                echo ""
                echo "💡 Try pulling first: git pull origin main --rebase"
                echo "   Then push: git push -u origin main"
            fi
        fi
    fi

    echo ""
    echo "✨ Done! Remote configured."
    exit 0
fi

# No URL provided - offer to create new repository
echo ""
echo "📝 No URL provided. Let's create a new repository!"
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "🎯 GitHub CLI detected!"
    echo ""
    read -p "❓ Create new repository with GitHub CLI? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        read -p "Enter repository name (default: planet-pipeline): " repo_name
        repo_name=${repo_name:-planet-pipeline}

        echo ""
        read -p "Make repository public? (y/n, default: y) " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Nn]$ ]]; then
            visibility="--private"
        else
            visibility="--public"
        fi

        echo ""
        echo "📤 Creating GitHub repository and pushing..."
        gh repo create "$repo_name" $visibility --source=. --remote=origin --push

        echo ""
        echo "🎉 Repository created and pushed!"
        echo "🔗 View at: https://github.com/$(gh api user -q .login)/$repo_name"
        exit 0
    fi
fi

# Manual setup instructions
echo ""
echo "ℹ️  Manual Setup Instructions:"
echo ""
echo "1️⃣  Go to: https://github.com/new"
echo "2️⃣  Repository name: planet-pipeline"
echo "3️⃣  Description: Production-ready Planet Labs satellite imagery processing pipeline"
echo "4️⃣  Choose Public or Private"
echo "5️⃣  Do NOT initialize with README, .gitignore, or license"
echo "6️⃣  Click 'Create repository'"
echo ""
echo "Then run these commands:"
echo ""
echo "  git remote add origin https://github.com/YOUR_USERNAME/planet-pipeline.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""

echo ""
echo "✨ Done!"
