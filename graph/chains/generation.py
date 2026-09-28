from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(temperature=0)


# Inline equivalent of hub.pull("rlm/rag-prompt") - the hub module was removed
# from the langchain package, and pulling public prompts now requires a Langsmith API key

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you dont know the answer, just say that you dont know. "
            "Use three sentences maximum and keep the answer concise. \n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

generation_chain = prompt | llm | StrOutputParser()
