from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os   

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM APP'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model = 'gpt-4o-mini', temperature=0.7)
model2 = ChatOpenAI(model = 'gpt-4o', temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'Sequential Chain Run',
    'tags': [
        'llm app',
        'report generation',
        'summarization',
    ],
    'metadata': {
        'model1': 'gpt-4o-mini',
        'model1_temp': 0.7,
        'parser': 'stroutputparser',
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
