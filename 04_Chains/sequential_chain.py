import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI( 
  model=os.getenv("MODEL", ""),
  base_url=os.getenv("API_URL", ""),
  api_key=SecretStr(os.getenv("API_KEY", "")),
  temperature=0
)  # replace with your LLM

# 1️⃣ Step 1: Review Prompt
prompt_review = PromptTemplate.from_template(
    "You ordered {dish_name} and your experience was {experience}. Write a review:"
)

# 2️⃣ Step 2: Follow-up Comment Prompt
prompt_comment = PromptTemplate.from_template(
    "Given the restaurant review: {review}, write a follow-up comment:"
)

# 3️⃣ Step 3: Summary Prompt
prompt_summary = PromptTemplate.from_template(
    "Summarise the review in one short sentence:\n\n {comment}"
)

# 4️⃣ Step 4: Translation Prompt
prompt_translation = PromptTemplate.from_template(
    "Translate the summary to Bengali:\n\n {summary}"
)

# Runnables
parse_output = StrOutputParser()

# Review Chain
review_chain = (
  # Step 1: Review
  prompt_review 
    | llm 
    | parse_output
    | RunnableLambda(lambda x: {"review": x})
)

comment_chain = (
  # Step 2: Comment
  prompt_comment
    | llm
    | parse_output
    | RunnableLambda(lambda x: {"comment": x})
)

summary_chain = (
  # Step 3: Summary 
  prompt_summary
    | llm
    | parse_output
    | RunnableLambda(lambda x: {"summary": x})
)

translation_chain = (
  # Step 4: Translation
  prompt_translation
    | llm
    | parse_output
)

# LCEL Pipeline
# chain = (
#   review_chain 
#   | comment_chain 
#   | summary_chain 
#   | translation_chain
# )

# Sequential LCEL chain
chain = (
    RunnableMap({
        "review": {"dish_name": RunnablePassthrough(), "experience": RunnablePassthrough()} 
                  | prompt_review | llm | parse_output,
    })
    | RunnableMap({
        "comment": {"review": RunnablePassthrough()} | prompt_comment | llm | parse_output,
        "review": RunnablePassthrough(),
    })
    | RunnableMap({
        "summary": {"comment": RunnablePassthrough()} | prompt_summary | llm | parse_output,
        "review": RunnablePassthrough(),
        "comment": RunnablePassthrough(),
    })
    | RunnableMap({
        "bengali_translation": {"summary": RunnablePassthrough()} | prompt_translation | llm | parse_output,
        "review": RunnablePassthrough(),
        "comment": RunnablePassthrough(),
        "summary": RunnablePassthrough(),
    })
)

# Run
result = chain.invoke({"dish_name": "Pizza Salami", "experience": "It was awful!"})
print(result)

# # Run the chain
# result = chain.invoke({"dish_name": "Pizza Salami", "experience": "It was awful!"})
# print(result)
