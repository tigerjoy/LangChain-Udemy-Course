import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", ""),
    base_url=os.getenv("API_URL", ""),
    api_key=SecretStr(os.getenv("API_KEY", "")),
    temperature=0.7
)

parse_output = StrOutputParser()

# Prompts for each platform
prompt_instagram = PromptTemplate.from_template(
    "Create a fun and engaging Instagram caption for this product: {product_description}"
)

prompt_twitter = PromptTemplate.from_template(
    "Write a witty, under-280-character tweet about: {product_description}"
)

prompt_linkedin = PromptTemplate.from_template(
    "Write a professional LinkedIn post promoting: {product_description}"
)

# Runnable chains for each platform
instagram_chain = prompt_instagram | llm | parse_output
twitter_chain = prompt_twitter | llm | parse_output
linkedin_chain = prompt_linkedin | llm | parse_output

# Run them in parallel
social_media_chain = RunnableParallel(
    instagram=instagram_chain,
    twitter=twitter_chain,
    linkedin=linkedin_chain
)

# Example run
result = social_media_chain.invoke({
    "product_description": "A new eco-friendly reusable water bottle that keeps drinks cold for 24 hours."
})

print(result)
