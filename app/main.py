import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.router as router

root_path = os.environ.get("ROOT_PATH", "")
app = FastAPI(
    title="LangSync",
    version="1.0",
    # proxy stuff (don't forget to run app with "--proxy-headers")
    root_path=root_path,
    root_path_in_servers=not root_path,
    servers=[{'url': '.'}],
    redoc_url=None
)
app.router.redirect_slashes = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router.router)
