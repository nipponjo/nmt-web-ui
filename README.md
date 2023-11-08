# nmt-web-ui

A local Web UI for the [huggingface/Helsinki-NLP](https://huggingface.co/Helsinki-NLP) neural machine translation models.

Run with:
```bash
python app.py
```
Preview:

<img src="https://github.com/nipponjo/nmt-web-ui/assets/28433296/3b00b54c-01c3-44da-b5a5-a3efa45481f5" width="80%"></img>

dark mode:

<img src="https://github.com/nipponjo/nmt-web-ui/assets/28433296/82296fea-8c31-4e9e-be86-400736c69bd2" width="80%"></img>

display cross-attention values:

<img src="https://github.com/nipponjo/nmt-web-ui/assets/28433296/1b89a965-628e-47fd-80fd-819ac945133f" width="78%"></img>


Requirements:

- PyTorch
- [huggingface/transformers](https://huggingface.co/docs/transformers/installation)
- [FastAPI](https://fastapi.tiangolo.com/): for the backend api | uvicorn: for serving the app
- Install with: `pip install fastapi "uvicorn[standard]"`
