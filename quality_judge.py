import os

from openai import AzureOpenAI

tweets = [
    "Biden has no plan to control the border. His reckless immigration policies are pushing families toward human smuggling and making it more likely that more people will die trying to get to America.",
    "Our democracy is at stake and we need to act now to protect our democracy.",
    "I was thinking about an"
]

# from pydantic import BaseModel
# class QualityScore(BaseModel):
#     score: int

client = AzureOpenAI(
    azure_endpoint=os.getenv(f"AZURE_OPENAI_ENDPOINT_gpt_4o_mini"),
    api_key=os.getenv("AZURE_OPENAI_KEY_gpt_4o_mini"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_gpt_4o_mini"),
)

for tweet in tweets:
    print("tweet:", tweet)
    prompt = "You are a text quality judge.\n" + \
             "When judging the quality pay attention to:\n" + \
             "\t- broken/cut-off text\n" + \
             "\t- very repetitive text\n" + \
             "\t- grammar\n" + \
             "\t- semantic plausability\n" + \
             "\t- lexical complexity\n" + \
             "\nJudge the quality of a given internet post and reply ONLY with a integer from 0-2.\n" + \
             "\t0 - low quality\n\t1 - intermediate quality\n\t2 - good quality\n\n" + \
             f"Here is the post: {tweet}"

    # completion = client.beta.chat.completions.parse(
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0000001,
        messages=[
            {"role": "system", "content": prompt},
        ],
    )
    # score = completion.choices[0].message.parsed
    print(completion.choices[0].message.content)
    print(completion.usage.total_tokens)
    from IPython import embed; embed();

# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# completion = client.chat.completions.create(
#     model="gpt-4o-mini-2024-07-18",
#     temperature=0.0000001,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
# )
#
# print(completion.choices[0].message.content)
