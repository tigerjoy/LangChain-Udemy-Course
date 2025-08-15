import os
import sys
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL", ""),
    base_url=os.getenv("API_URL", ""),
    api_key=SecretStr(os.getenv("API_KEY", "")),
    temperature=0
)

parse_output = StrOutputParser()

class Classification(BaseModel):
    category: Literal["complaint", "question", "praise", "unknown"] = Field(
        description="Category of the message. One of: complaint, question, praise, unknown"
    )

# Step 1: Classifier prompt
classification_prompt = PromptTemplate.from_template(
    """
      Classify the customer message into exactly one of: complaint, question, praise or unknown.
      
      Message: {message}
  
    """
)

classification_chain = classification_prompt | llm.with_structured_output(Classification)

# Step 2: Response generators
complaint_chain = PromptTemplate.from_template(
    "This is a customer complaint: {message}. Write an empathetic apology and say support will contact them."
) | llm | parse_output

question_chain = PromptTemplate.from_template(
    "This is a customer question: {message}. Answer it politely and clearly."
) | llm | parse_output

praise_chain = PromptTemplate.from_template(
    "This is positive feedback: {message}. Write a warm thank-you response."
) | llm | parse_output

# Step 3: Conditional branching
triage_chain = classification_chain | RunnableBranch(
    # If category contains 'complaint'
    (lambda c: c.category.lower() == "complaint", complaint_chain),
    # If category contains 'question'
    (lambda c:  c.category.lower() == "question", question_chain),
    # If category contains 'praise'
    (lambda c:  c.category.lower() == "praise", praise_chain),
    # Default case
    lambda x: f"Sorry, I could not determine the category for: {x}"
)

# Example runs
messages = [
    "I'm really unhappy, my order arrived late and the product was damaged.",
    "What's the warranty period for your coffee machine?",
    "I love your products! The new packaging looks amazing.",
    "My name is 67W"
]

for m in messages:
    result = triage_chain.invoke({"message": m})
    print(f"Message: {m}\nResponse: {result}\n{'-'*60}")
