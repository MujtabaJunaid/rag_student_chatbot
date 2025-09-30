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
        cohere_key = os.environ.get("default")
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
        return self.vector_store.as_retriever()

vector_manager = VectorStoreManager()

class IntentAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Analyze this student query and return ONLY valid JSON with intent and filters:
            Query: "{query}"
            
            Return JSON format exactly like this:
            {{
                "intent": "performance_review",
                "subject_filter": null,
                "topic_filter": null
            }}
            
            Possible intents: performance_review, improvement_tips, subject_analysis, general_advice
            """
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="Llama 3.3 70B",
                temperature=0.1
            )
            response_text = chat_completion.choices[0].message.content.strip()
            return json.loads(response_text)
        except Exception as e:
            return {"intent": "performance_review", "subject_filter": None, "topic_filter": None}

class PerformanceAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_performance(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"analysis": "No performance data found for this student ID. Please make sure you have added question data first.", "documents": documents}
        
        doc_texts = [doc.page_content for doc in documents]
        context = "\n\n".join(doc_texts[:3])
        
        try:
            prompt = f"""
            Based on the following student performance data, provide a brief analysis of strengths and weaknesses:
            
            {context}
            
            Keep the analysis concise and focused on key patterns.
            """
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-70b-versatile",
                temperature=0.1
            )
            return {"analysis": chat_completion.choices[0].message.content, "documents": documents}
        except Exception as e:
            return {"analysis": f"Performance analysis completed. Found {len(documents)} relevant records.", "documents": documents}

class Advisor:
    def __init__(self):
        self.client = client

    def generate_advice(self, analysis: Dict[str, Any], intent: str, query: str, student_id: str) -> str:
        try:
            prompt = f"""
            Student ID: {student_id}
            Student Question: {query}
            
            Performance Analysis:
            {analysis['analysis']}
            
            Provide helpful, actionable advice based on the student's question and performance data.
            If no specific performance data is available, provide general study advice.
            """
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="Llama 3.3 70B",
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return "Based on your performance data, I recommend focusing on consistent practice and reviewing feedback from your assessments."

class StudentChatbot:
    def __init__(self):
        self.intent_analyzer = IntentAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.advisor = Advisor()
        self.vector_manager = vector_manager

    def process_query(self, student_id: str, query: str) -> Dict[str, Any]:
        try:
            print(f"Processing query for student {student_id}: {query}")
            
            intent_data = self.intent_analyzer.analyze_intent(query)
            print(f"Intent analysis: {intent_data}")
            
            retriever = self.vector_manager.get_retriever()
            filters = {"student_id": student_id}
            documents = retriever.get_relevant_documents(query, filter=filters)
            print(f"Retrieved {len(documents)} documents")
            
            performance_analysis = self.performance_analyzer.analyze_performance(documents)
            print("Performance analysis completed")
            
            advice = self.advisor.generate_advice(
                performance_analysis, 
                intent_data.get("intent", "performance_review"), 
                query, 
                student_id
            )
            print("Advice generated")
            
            return {
                "response": advice,
                "analysis": performance_analysis["analysis"],
                "documents_used": len(documents)
            }
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Please try again with a different question or make sure you have added your performance data first.",
                "analysis": f"Technical issue: {str(e)}",
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
            "response": "I apologize for the inconvenience. Please try your question again or contact support if the issue persists.",
            "analysis": "Service temporarily unavailable. Please try again.",
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
