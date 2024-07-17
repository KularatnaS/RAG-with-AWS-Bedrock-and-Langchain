import boto3
import os
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock

knowledge_base_id = "AWS_KNOWLEDGE_BASE_ID"
aws_profile = "your-profile"

os.environ["AWS_PROFILE"] = aws_profile
# Create a new Bedrock client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

modelID = "anthropic.claude-v2"

llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000,"temperature":0.9}
)

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=knowledge_base_id,
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

def rag_chat(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True)
    return qa(query)

if __name__ == "__main__":
    response = rag_chat("In what subject does Shashitha have a PhD?")
    print(response['result'])
