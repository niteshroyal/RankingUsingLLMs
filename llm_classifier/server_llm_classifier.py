import os
import torch
import socket
import logging
from flask import Flask, request, jsonify

from conf import configuration

from llm_classifier.eval_classifier import EvalLargeLanguageModel


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


app = Flask(__name__)


class LLMClassifierServer(EvalLargeLanguageModel):

    def __init__(self):
        super().__init__()
        self.init_ft_model()

    def evaluate(self, eval_prompts, max_new_tokens=configuration.tokenizer_max_length):
        model_input = self.tokenizer(
            eval_prompts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            eval_logits = eval_outputs.logits
            return eval_logits


@app.route('/debug', methods=['POST'])
def chat_completion_debug():
    data = request.json
    prompts = data["prompts"]
    llm_response = server.evaluate(prompts)
    llm_response_list = llm_response.cpu().tolist()
    response = {"llm_response": llm_response_list,
                "finetuned_llm_location": server.fine_tuned_model_location,
                "finetuned_llm": configuration.base_model_id,
                "prompt": prompts}
    return jsonify(response)


@app.route('/completion', methods=['POST'])
def chat_completion():
    data = request.json
    prompts = data["prompts"]
    llm_response = server.evaluate(prompts)
    llm_response_list = llm_response.cpu().tolist()
    response = {"llm_response": llm_response_list}
    return jsonify(response)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception as e:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


if __name__ == '__main__':
    initialization()
    server = LLMClassifierServer()
    ip_address = get_ip()
    port = 5000
    logging.info(f"Starting server at http://{ip_address}:{port}/completion")
    app.run(host='0.0.0.0', port=port, debug=False)
