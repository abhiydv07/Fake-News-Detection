# How to Deploy Fake News Detection API

## Option 1: Deploy to Render (Free, Recommended)

### Step 1: Upload to GitHub
1. Go to https://github.com/new
2. Create a new repository called `fake-news-detection`
3. Upload all files from the `fake-news-detection` folder

### Step 2: Deploy on Render
1. Go to https://render.com and sign up (free)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Settings:
   - **Name**: fake-news-detection
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api.py`
5. Click **Create Web Service**
6. Wait 2-3 minutes for deployment

### Step 3: Share the link
Your API will be live at: `https://fake-news-detection.onrender.com`

Share this link with anyone. They can:
- Open `https://fake-news-detection.onrender.com/docs` for the interactive API docs
- Use the API from any app or website

---

## Option 2: Deploy to Railway (Free tier)

1. Go to https://railway.app and sign up
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your repository
4. Railway auto-detects Python and deploys
5. Your API will be live at the provided URL

---

## Option 3: Run on Your Computer (Share via ngrok)

### Step 1: Install ngrok
Download from https://ngrok.com and install

### Step 2: Start the API
```bash
cd fake-news-detection
python api.py
```

### Step 3: Share via ngrok
In a new terminal:
```bash
ngrok http 8000
```

This gives you a public URL like `https://abc123.ngrok.io` that anyone can access.

---

## What People Can Do With Your API

Once deployed, anyone can:

1. **Test via browser**: Open `your-url/docs` for interactive API docs
2. **Use from code**:
```python
import requests
resp = requests.post("your-url/predict", json={"text": "Some news headline"})
print(resp.json())
```
3. **Batch classify**: Send up to 1000 articles at once
4. **Build apps**: Use the API in websites, mobile apps, or scripts
