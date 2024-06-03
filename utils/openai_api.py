import os
import time
import openai
import logging

from conf import configuration

# ELEXIR API key
os.environ['OPENAI_API_KEY'] = 'hidden'
openai.api_key = os.getenv('OPENAI_API_KEY')


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def token_size_determination(prompt):
    print(prompt)
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1
    )
    token_count = response['usage']['total_tokens']
    print("Number of tokens:", token_count)


class OpenAI:
    def __init__(self, model=None):
        self.model = model
        # self.model = "gpt-3.5-turbo"
        # self.model = "text-davinci-003"
        # self.model = "gpt-4"
        # self.model = "gpt-4-0613"
        # self.model = "gpt-4-1106-preview"
        # self.model = "gpt-3.5-turbo-0613"
        # self.model = "text-davinci-003"
        # self.model = "gpt-3.5-turbo-instruct"

    def get_gpt_response(self, prompt):
        processed = False
        response = None
        while not processed:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                processed = True
            except openai.error.ServiceUnavailableError as e:
                logging.info(f"OpenAI API service is unavailable: {e}")
                print(f"OpenAI API service is unavailable: {e}")
                time.sleep(300)
            except openai.error.APIError as e:
                logging.info(f"OpenAI API returned an API Error: {e}")
                print(f"OpenAI API returned an API Error: {e}")
            except openai.error.APIConnectionError as e:
                logging.info(f"Failed to connect to OpenAI API: {e}")
                print(f"Failed to connect to OpenAI API: {e}")
            except openai.error.RateLimitError as e:
                logging.info(f"OpenAI API request exceeded rate limit: {e}")
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(60)
            except openai.error.Timeout as e:
                logging.info(f"Request timed out: {e}")
                print(f"Request timed out: {e}")
                time.sleep(120)
        message = response.choices[0].message.content.strip()
        return message

    # def get_gpt_response(self, prompt):
    #     processed = False
    #     response = None
    #     while not processed:
    #         try:
    #             response = openai.Completion.create(
    #                 model=self.model,
    #                 prompt=prompt,
    #                 max_tokens=150
    #             )
    #             processed = True
    #         except openai.error.APIError as e:
    #             print(f"OpenAI API returned an API Error: {e}")
    #         except openai.error.APIConnectionError as e:
    #             print(f"Failed to connect to OpenAI API: {e}")
    #         except openai.error.RateLimitError as e:
    #             print(f"OpenAI API request exceeded rate limit: {e}")
    #             time.sleep(60)
    #         except openai.error.Timeout as e:
    #             print(f"Request timed out: {e}")
    #             time.sleep(120)
    #     message = response.choices[0].text.strip()
    #     return message


if __name__ == '__main__':
    initialization()
    token_size_determination('ABC')
    # extractor = OpenAI()
