# Multimodal Fashion Search

Run the app locally with Docker:

```bash
docker build -t fashion-app .
docker run -p 8000:8000 fashion-app
```

Open http://localhost:8000 in a browser.

Product images are served from `static/images/`. Copy the dataset image files into
that directory (for example, `static/images/15970.jpg`) so they can be displayed
by the search results.
![Fashion Search UI](screenshots/screencapture-localhost-8000-2026-08-24-105229.png)