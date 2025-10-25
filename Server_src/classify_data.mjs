import { pipeline, AutoProcessor, AutoModelForImageTextToText, load_image, TextStreamer } from "@huggingface/transformers";
import * as fs from "fs";
import * as path from 'path';
const model = "onnx-community/FastVLM-0.5B-ONNX";
const processor = await AutoProcessor.from_pretrained(model);
const imageTextToTextModel = await AutoModelForImageTextToText.from_pretrained(model, {
  dtype: "q4",
});

//const extractor = await pipeline("feature-extraction", "model_q4")


const message = {
    role: "user",
    content: "<image>STATE THE ARTICLE OF CLOTHING IN THE IMAGE. SHORT.",
}
const prompt = processor.apply_chat_template([message], {
    add_generation_prompt: true,
});
let img = await load_image(path.resolve("./static/dataset/train/hat/00d94e21-5891-492e-be0e-792e7338c077.jpg"))
console.log("Image loaded");
let inputs = await processor(img, prompt, {
    add_special_tokens: false,
});

const output = await imageTextToTextModel.generate({
    ...
    inputs,
    max_new_tokens: 32,
    do_sample: false,
    streamer: new TextStreamer(processor.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
    }),
});

const decoded = processor.batch_decode(
  outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
  { skip_special_tokens: true },
);
console.log(decoded[0]);