import express from "express";
import { LlamaCpp } from "node-llama-cpp";
import { pipeline, AutoProcessor, AutoModelForImageTextToText, load_image, TextStreamer } from "@huggingface/transformers";
const model = "vision_encoder_q4f16";
const processor = await AutoProcessor.from_pretrained(model);
const imageTextToTextModel = await AutoModelForImageTextToText.from_pretrained(model, {
  dtype: {
    embed_tokens: "fp16",
    vision_encoder: "q4",
    decoder_modal_meged: "q4",
  },
});
const app = express();
const port = 8080;

app.get("/", (req, res) => {
  res.send("Hello World!");
});

