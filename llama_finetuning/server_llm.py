import os
import torch
import logging
import socket
from flask import Flask, request, jsonify

from conf import configuration
from llama_finetuning.test_finetuned_llm import TestLargeLanguageModel


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


app = Flask(__name__)


class LLMServer(TestLargeLanguageModel):

    def __init__(self):
        super().__init__()
        self.init_ft_model()

    def evaluate(self, eval_prompt, max_new_tokens=configuration.max_new_tokens):
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            result = self.tokenizer.decode(self.ft_model.generate(**model_input,
                                                                  max_new_tokens=max_new_tokens,
                                                                  pad_token_id=2)[0], skip_special_tokens=True)
        return result


@app.route('/completion', methods=['POST'])
def chat_completion():
    data = request.json
    prompt = data["prompt"]
    max_new_tokens = data["max_new_tokens"]
    llm_response = server.evaluate(prompt, max_new_tokens)
    response = {"llm_response": llm_response,
                "finetuned_llm_location": server.fine_tuned_model_location,
                "finetuned_llm": configuration.base_model_id,
                "prompt": prompt}
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
    server = LLMServer()
    ip_address = get_ip()
    port = 5000
    logging.info(f"Starting server at http://{ip_address}:{port}/completion")
    app.run(host='0.0.0.0', port=port, debug=False)
