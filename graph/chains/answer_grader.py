from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """ You are a grader assessing whether an LLM generation addresses / resolves the question. \n
            Give a binary score 'yes' or 'no'. 'yes' means the answer addresses / resolves the question. """

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Userquestion: {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader