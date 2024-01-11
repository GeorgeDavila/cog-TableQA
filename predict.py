# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import pipeline
import pandas as pd
    
TASK_CLASS = "table-question-answering"
MODEL_NAME = "google/tapas-large-finetuned-wtq"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=TOKEN_CACHE
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        model.generation_config = GenerationConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.model = model.to("cuda")

    def predict(
        self,
        query: str = Input(
            description="Your question.", 
            default="What age was Charles Alexander Fortune?"
            ),
        userFileType: str = Input(
            default="csv",
            choices=["csv", "excel", "json"],
            description="File type",
        ),
        userFile: Path = Input(
            description="Upload a file", 
            default=Path("/titanic.csv")
            ),
    ) -> str:
        """Run a single prediction on the model"""

        pipe = pipeline(TASK_CLASS, model=MODEL_NAME)

        if userFileType == "csv":
            data = pd.read_csv(userFile)
        if userFileType == "excel":
            data = pd.read_excel(userFile)
        if userFileType == "json":
            data = pd.read_json(userFile)
        if userFileType == "sql":
            data = pd.read_sql(userFile)
        if userFileType == "html":
            data = pd.read_html(userFile)

        response = pipe(table=data, query=query)
        print(response)

        return response
