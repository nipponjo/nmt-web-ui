# nmt-web-ui

A local Web UI for the [huggingface/Helsinki-NLP](https://huggingface.co/Helsinki-NLP) neural machine translation models.

Run with:
```bash
python app.py
```
Preview:

![image](https://github.com/nipponjo/nmt-web-ui/assets/28433296/6314ce80-797f-4642-acbc-65c7139bc760)

Requirements:

- PyTorch
- [huggingface/transformers](https://huggingface.co/docs/transformers/installation)
- [FastAPI](https://fastapi.tiangolo.com/): for the backend api | uvicorn: for serving the app
- Install with: `pip install fastapi "uvicorn[standard]"`
