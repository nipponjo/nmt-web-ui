# nmt-web-ui

A local Web UI for the [huggingface/Helsinki-NLP](https://huggingface.co/Helsinki-NLP) neural machine translation models.

Run with:
```bash
python app.py
```
Preview:

<img src="https://github.com/nipponjo/nmt-web-ui/assets/28433296/5334b131-4684-4fc6-9ba3-98257ac1fcca" width="80%"></img>

dark mode:

<img src="https://github.com/nipponjo/nmt-web-ui/assets/28433296/7b08b229-f2b3-427d-a05a-b645219c24cb" width="80%"></img>

Requirements:

- PyTorch
- [huggingface/transformers](https://huggingface.co/docs/transformers/installation)
- [FastAPI](https://fastapi.tiangolo.com/): for the backend api | uvicorn: for serving the app
- Install with: `pip install fastapi "uvicorn[standard]"`
