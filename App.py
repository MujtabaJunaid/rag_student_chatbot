import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from groq import Groq

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langgraph.graph import Graph, END
from chromadb.config import Settings as ChromaSettings
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("groq_api_key"))

class QuestionData(BaseModel):
    student_id: str
    assessment_id: str
    question: str
    question_type: str
    total_marks: int
    obtained_marks: int
    feedback: str
    tags: List[str]
    date: str

class ChatQuery(BaseModel):
    student_id: str
    query: str

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        cohere_key = os.environ.get("COHERE_API_KEY")
        self.embeddings = None
        if cohere_key:
            try:
                self.embeddings = CohereEmbeddings(cohere_api_key=cohere_key)
            except Exception:
                self.embeddings = None
        self.vector_store = None
        try:
            settings = ChromaSettings(
                persist_directory=persist_directory,
                chroma_db_impl="duckdb+parquet",
                anonymized_telemetry=False
            )
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                client_settings=settings
            )
        except Exception:
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception:
                self.vector_store = None

    def _ensure_vector_store(self):
        if self.vector_store is None:
            if self.embeddings is None:
                raise RuntimeError("Embeddings not configured; set COHERE_API_KEY environment variable.")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Chroma vector store: {e}")

    def add_question_data(self, question_data: QuestionData):
        self._ensure_vector_store()
        doc_text = f"""
        Student: {question_data.student_id}
        Assessment: {question_data.assessment_id}
        Question: {question_data.question}
        Type: {question_data.question_type}
        Marks: {question_data.obtained_marks}/{question_data.total_marks}
        Feedback: {question_data.feedback}
        Tags: {','.join(question_data.tags)}
        Date: {question_data.date}
        """
        metadata = {
            "student_id": question_data.student_id,
            "assessment_id": question_data.assessment_id,
            "question_type": question_data.question_type,
            "total_marks": question_data.total_marks,
            "obtained_marks": question_data.obtained_marks,
            "tags": ','.join(question_data.tags),
            "date": question_data.date,
            "feedback": question_data.feedback
        }
        doc = Document(page_content=doc_text, metadata=metadata)
        self.vector_store.add_documents([doc])

    def get_retriever(self):
        self._ensure_vector_store()
        return self.vector_store.as_retriever(search_kwargs={"k": 1})

vector_manager = VectorStoreManager()

class StudentChatbot:
    def __init__(self):
        self.client = client
        self.vector_manager = vector_manager

    def process_query(self, student_id: str, query: str) -> Dict[str, Any]:
        try:
            print(f"Processing query for student {student_id}: {query}")
            
            retriever = self.vector_manager.get_retriever()
            filters = {"student_id": student_id}
            documents = retriever.invoke(query, filter=filters)
            print(f"Retrieved {len(documents)} documents")
            
            if not documents:
                return {
                    "response": "No performance data found for your student ID. Please add your assessment data first.",
                    "analysis": "No documents found for this student ID.",
                    "documents_used": 0
                }
            
            doc_content = documents[0].page_content
            
            analysis_prompt = f"""
            Analyze this student's performance data:
            
            {doc_content}
            
            Provide a brief analysis focusing on performance patterns.
            """
            
            analysis_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama3-70b-8192",
                temperature=0.7
            )
            analysis = analysis_completion.choices[0].message.content
            
            advice_prompt = f"""
            Student Query: {query}
            
            Performance Data:
            {doc_content}
            
            Performance Analysis:
            {analysis}
            
            Provide specific, personalized advice addressing the student's query.
            """
            
            advice_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": advice_prompt}],
                model="llama3-70b-8192",
                temperature=0.7
            )
            advice = advice_completion.choices[0].message.content
            
            return {
                "response": advice,
                "analysis": analysis,
                "documents_used": len(documents)
            }
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Please try again.",
                "analysis": f"Error: {str(e)}",
                "documents_used": 0
            }

chatbot = StudentChatbot()

@app.post("/api/add_question_data")
async def add_question_data(question_data: QuestionData):
    try:
        vector_manager.add_question_data(question_data)
        return {"status": "success", "message": "Question data added to vector store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_bot(chat_query: ChatQuery):
    try:
        result = chatbot.process_query(chat_query.student_id, chat_query.query)
        return {
            "response": result["response"],
            "analysis": result["analysis"],
            "documents_used": result["documents_used"]
        }
    except Exception as e:
        return {
            "response": "I apologize for the inconvenience. Please try your question again.",
            "analysis": "Service temporarily unavailable.",
            "documents_used": 0
        }

@app.get("/")
async def root():
    return {"message": "Student Chatbot API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
