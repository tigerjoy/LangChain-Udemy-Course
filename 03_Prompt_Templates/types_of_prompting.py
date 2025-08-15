import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from pydantic import SecretStr, BaseModel, Field

load_dotenv()

model = ChatOpenAI(
  model=os.getenv("MODEL", ""),
  base_url=os.getenv("API_URL", ""),
  api_key=SecretStr(os.getenv("API_KEY", "")),
  temperature=0
)

class AnalysisResult(BaseModel):
  """The result of analysing the review into sentiment and subject"""
  review: str = Field(description="The review being analysed.")
  sentiment: Literal["positive", "neutral", "negative"] = Field(description="The sentiment of the review, positive, neutral or negative")
  subject: str = Field(description="The subject of the review")

def sentiment_analysis(TEMPLATE, review:str): 
  prompt_template = PromptTemplate(
    template=TEMPLATE,
    input_variables=["input"]
  )

  prompt = prompt_template.format(input=review)

  structured_model = model.with_structured_output(AnalysisResult)
  response = structured_model.invoke(prompt)

  return response

if __name__ == "__main__":

  reviews = [
    "I absolutely loved the movie, fantastic plot!",
    "It was okay, nothing special.",
    "I hated the movie it was so boring.",
    "This movie is full of plot holes."
  ]

  ZERO_SHOT_PROMPT_TEMPLATE = '''
    Interprete The review and evaluate the review sentiment.
    sentiment: Is the review in a positive, netural or negative sentiment?
    subject: What subject is the review about? Use exactly one word.

    Format the output as JSON with the following keys:
    sentiment
    subject
    review

    review: {input}
  '''

  FEW_SHOT_PROMPT_TEMPLATE = """
    Interprete the review and evaluate the review.
    sentiment: is the review in a positive, neutral or negative sentiment?
    subject: What subject is the review about? Use exactly one word.

    Format the output as JSON with the following keys:
    review
    sentiment
    subject

    review: {input}

    Examples:
    review: The BellaVista restaurant offers an exquisite dining experience. The flavors are rich and the presentation is impeccable.
    sentiment: positive
    subject: BellaVista

    review: BellaVista restaurant was alright. The food was decent, but nothing stood out.
    sentiment: neutral
    subject: BellaVista

    review: I was disappointed with BellaVista. The service was slow and the dishes lacked flavor.
    sentiment: negative
    subject: BellaVista

    review: SeoulSavor offered the most authentic Korean flavors I've tasted outside of Seoul. The kimchi was perfectly fermented and spicy.
    sentiment: positive
    subject: SeoulSavor

    review: SeoulSavor was okay. The bibimbap was good but the bulgogi was a bit too sweet for my taste.
    sentiment: neutral
    subject: SeoulSavor

    review: I didn't enjoy my meal at SeoulSavor. The tteokbokki was too mushy and the service was not attentive.
    sentiment: negative
    subject: SeoulSavor

    review: MunichMeals has the best bratwurst and sauerkraut I've tasted outside of Bavaria. Their beer garden ambiance is truly authentic.
    sentiment: positive
    subject: MunichMeals

    review: MunichMeals was alright. The weisswurst was okay, but I've had better elsewhere.
    sentiment: neutral
    subject: MunichMeals

    review: I was let down by MunichMeals. The potato salad lacked flavor and the staff seemed uninterested.
    sentiment: negative
    subject: MunichMeals
  """

  for review in reviews:
    response = sentiment_analysis(
      FEW_SHOT_PROMPT_TEMPLATE,
      review
    )
    print(repr(response))