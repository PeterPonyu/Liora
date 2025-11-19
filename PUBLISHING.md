# Publishing Liora to PyPI and GitHub

## Pre-publish Checklist

- [x] Package structure reorganized (lowercase `liora/`)
- [x] README.md with comprehensive documentation
- [x] LICENSE file (MIT)
- [x] pyproject.toml with all metadata
- [x] CHANGELOG.md
- [x] CONTRIBUTING.md
- [x] .gitignore
- [x] MANIFEST.in
- [x] Tests passing
- [x] GitHub Actions CI/CD workflows

## Local Testing

### 1. Test Import
```bash
cd /home/zeyufu/LAB/Liora
python -c "from liora import Liora; print('OK')"
```

### 2. Run Tests
```bash
# If pytest not installed:
pip install pytest pytest-cov

# Run tests
pytest -v
```

### 3. Run Examples
```bash
python -m liora.examples.encoder_example
```

### 4. Build Package
```bash
pip install build
python -m build
```

This creates `dist/` with `.whl` and `.tar.gz` files.

### 5. Test Install from Local Build
```bash
pip install dist/liora-0.4.0-py3-none-any.whl
```

## GitHub Setup

### 1. Initialize Git Repository
```bash
cd /home/zeyufu/LAB/Liora
git init
git add .
git commit -m "Initial commit: Liora v0.4.0"
```

### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Name: `liora`
3. Description: "Lorentz Information ODE-Regularized VAE for scRNA-seq"
4. Public or Private (recommend Public for PyPI)
5. **Don't** initialize with README (we have one)

### 3. Push to GitHub
```bash
git remote add origin https://github.com/PeterPonyu/liora.git
git branch -M main
git push -u origin main
```

### 4. Verify URLs in pyproject.toml
Ensure URLs point to your repo:
- `project.urls.Homepage`
- `project.urls.Repository`
- `project.urls.Issues`
Also verify README installation links and citation URL.

## PyPI Publishing

### 1. Create PyPI Account
- Go to https://pypi.org/account/register/
- Verify email

### 2. Create API Token
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name: "liora-upload"
5. Scope: "Entire account" (or specific to `liora` after first upload)
6. **Save the token** - you'll only see it once!

### 3. Configure Twine
```bash
pip install twine
```

Option A: Use token directly
```bash
twine upload dist/* -u __token__ -p pypi-YOUR_TOKEN_HERE
```

Option B: Save in ~/.pypirc
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

Then:
```bash
twine upload dist/*
```

### 4. Verify Upload
After upload, check:
- https://pypi.org/project/liora/

### 5. Test Install from PyPI
```bash
pip install liora
python -c "from liora import Liora; print('PyPI install OK')"
```

## GitHub Actions Setup (Automated Publishing)

### 1. Add PyPI Token to GitHub Secrets
1. Go to your repo: `https://github.com/YOUR_USERNAME/liora`
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Your PyPI token (starting with `pypi-`)
6. Click "Add secret"

### 2. Automated Publishing on Release
The `.github/workflows/publish.yml` will automatically:
- Build the package
- Upload to PyPI

When you create a GitHub release.

### 3. Create a Release
```bash
git tag v0.4.0
git push origin v0.4.0
```

Or via GitHub UI:
1. Go to "Releases"
2. Click "Create a new release"
3. Tag: `v0.4.0`
4. Title: `Liora v0.4.0`
5. Description: Copy from CHANGELOG.md
6. Click "Publish release"

GitHub Actions will automatically publish to PyPI!

## Post-Publication

### 1. Update README Badges
Add to README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/liora.svg)](https://badge.fury.io/py/liora)
[![Downloads](https://pepy.tech/badge/liora)](https://pepy.tech/project/liora)
```

### 2. Announce
- Post on relevant communities (r/MachineLearning, r/bioinformatics)
- Tweet about it
- Add to your CV/website

## Maintenance

### For Future Releases

1. Update version in `liora/__init__.py`
2. Update version in `pyproject.toml`
3. Update CHANGELOG.md
4. Commit changes
5. Create new tag and release
6. GitHub Actions will auto-publish

## Common Issues

### Import Error After Restructure
If you get import errors, ensure:
- Package name is lowercase: `liora/` not `Liora/`
- `__init__.py` exports `Liora` class
- Tests use `from liora import ...`

### Build Fails
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info
python -m build
```

### PyPI Upload Fails
- Check package name isn't taken: https://pypi.org/project/liora/
- Verify token is correct
- Ensure version number is incremented (can't re-upload same version)

## Quick Command Reference

```bash
# Local testing
python -m build
pip install dist/*.whl
python -c "from liora import Liora"

# Git
git add .
git commit -m "message"
git push

# PyPI
twine upload dist/*

# Tag release
git tag v0.3.0
git push origin v0.3.0
```

## Success Checklist

- [ ] Tests pass locally
- [ ] Package builds successfully
- [ ] Local install from wheel works
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] PyPI package uploaded
- [ ] PyPI install works: `pip install liora`
- [ ] GitHub release created
- [ ] Documentation updated
- [ ] CI/CD workflows passing
