# nmt-web-ui

A local Web UI for the [huggingface/Helsinki-NLP](https://huggingface.co/Helsinki-NLP) neural machine translation models.

Run with:
```bash
python app.py
```
Preview:

![image](https://github.com/nipponjo/nmt-web-ui/assets/28433296/d5860e63-c3ab-4d74-b6c5-96a2d73ca0d5)

Requirements:

- PyTorch
- [huggingface/transformers](https://huggingface.co/docs/transformers/installation)
- [FastAPI](https://fastapi.tiangolo.com/): for the backend api | uvicorn: for serving the app
- Install with: `pip install fastapi "uvicorn[standard]"`