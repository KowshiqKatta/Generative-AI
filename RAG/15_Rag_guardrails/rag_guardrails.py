from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from dotenv import load_dotenv
from guardrails_grhub_guardrails_pii import GuardrailsPII
from guardrails_grhub_toxic_language import ToxicLanguage
from guardrails_grhub_detect_jailbreak import DetectJailbreak
from tryolabs_grhub_restricttotopic import RestrictToTopic
from guardrails_grhub_response_evaluator import ResponseEvaluator
from guardrails import Guard, ValidationOutcome

# load the api keys
load_dotenv()

# create the validators for each component
pii_validator = GuardrailsPII(entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
toxic_language_validator = ToxicLanguage(on_fail="fix")
jailbreak_validator = DetectJailbreak(threshold=0.8, on_fail="exception")
restrict_to_topic_validator = RestrictToTopic(valid_topics=["AI", "Machine Learning", "Data Science", "Deep Learning"],
                                              invalid_topics=["Politics", "Religion", "Sports", "Entertainment"],
                                              on_fail="exception")
response_evaluator = ResponseEvaluator(llm_callable="gpt-5-mini",
                                       on_fail="reask")

input_validators = Guard().use(pii_validator,
                               toxic_language_validator,
                               restrict_to_topic_validator)

context_validators = Guard().use(jailbreak_validator)

response_validators = Guard().use(response_evaluator)

# build the llm
llm = ChatOpenAI(model="gpt-5-mini")

# graph state
class GuardState(TypedDict):
    
    original_input: str
    validated_input: str
    input_exception: bool
    
    original_context: str
    validated_context: str
    context_exception: bool
    
    original_response: str
    validated_response: str
    
    did_retrieve: bool    # dummy variable


# build the nodes
def validate_inputs(state: GuardState):
    user_input = state["original_input"]
    
    try:
        input_val_result: ValidationOutcome = input_validators.validate(user_input)
    except Exception as e:
        return {
            "input_exception": True,
            "validated_input": e}
    else:
        return {
            "input_exception": False,
            "validated_input": input_val_result.validated_output}


def validate_context(state: GuardState):
    retrieved_context = state["original_context"]

    try:
        context_val_result: ValidationOutcome = context_validators.validate(retrieved_context)
    except Exception as e:
        return {
            "context_exception": True,
            "validated_context": e}
    else:
        return {
            "context_exception": False,
            "validated_context": context_val_result.validated_output}


def retrieve(state: GuardState):
    return {
        "did_retrieve": True
    }


def response(state: GuardState):
    validated_context = state["validated_context"]
    query = state["validated_input"]
    
    # build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant and have knowledge about AI, Machine Learning, Data Science, and Deep Learning. Always answer questions using only the provided context."),
        ("system", "If you don't know the answer, say you don't know. Do not try to make up an answer."),
        ("system", "Here is the context: {context}"),
        ("user", "{query}")
    ])
    
    response = (prompt | llm).invoke({"context": validated_context, "query": query}).content
    
    return {"original_response": response}


def validate_response(state: GuardState):
    response = state["original_response"]
    user_query = state["validated_input"]
    
    response_val_result: ValidationOutcome = response_validators.validate(response,
                                                                          metadata={"validation_question": user_query})
    if not response_val_result.validation_passed:
        return {"validated_response": "The response is unable to answer the question based on the provided context. Please re-ask the question or provide more context."}
    else:
        return {"validated_response": response_val_result.validated_output}

# router functions
def exception_input_validation(state: GuardState) -> Literal["exception", "retrieve"]:
    if state["input_exception"]:
        return "exception"
    else:
        return "retrieve"

def exception_context_validation(state: GuardState) -> Literal["exception", "response"]:
    if state["context_exception"]:
        return "exception"
    else:
        return "response"

# build the graph
graph = StateGraph(GuardState)

# add nodes
graph.add_node("validate_inputs", validate_inputs)
graph.add_node("validate_context", validate_context)
graph.add_node("retrieve", retrieve)
graph.add_node("response", response)
graph.add_node("validate_response", validate_response)

# create edges
graph.add_edge(START, "validate_inputs")
graph.add_conditional_edges("validate_inputs", exception_input_validation, {"exception": END, "retrieve": "retrieve"})
graph.add_edge("retrieve", "validate_context")
graph.add_conditional_edges("validate_context", exception_context_validation, {"exception": END, "response": "response"})
graph.add_edge("response", "validate_response")
graph.add_edge("validate_response", END)

# compile the graph
compiled_graph = graph.compile()