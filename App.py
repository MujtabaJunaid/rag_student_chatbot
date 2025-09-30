import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from groq import Groq
import chromadb
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
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="student_performance")

    def add_question_data(self, question_data: QuestionData):
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
        doc_id = f"{question_data.student_id}_{question_data.assessment_id}_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[doc_text],
            metadatas=[{
                "student_id": question_data.student_id,
                "assessment_id": question_data.assessment_id,
                "question_type": question_data.question_type,
                "total_marks": question_data.total_marks,
                "obtained_marks": question_data.obtained_marks,
                "tags": ','.join(question_data.tags),
                "date": question_data.date,
                "feedback": question_data.feedback
            }],
            ids=[doc_id]
        )

    def search_documents(self, student_id: str, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"student_id": student_id}
        )
        return results

vector_manager = VectorStoreManager()

class StudentChatbot:
    def __init__(self):
        self.client = client
        self.vector_manager = vector_manager

    def call_groq_api(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.7,
                timeout=30
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                return "No response generated from AI."
        except Exception as e:
            return f"API Error: {str(e)}"

    def process_query(self, student_id: str, query: str) -> Dict[str, Any]:
        try:
            print(f"Processing query for student {student_id}: {query}")
            
            search_results = self.vector_manager.search_documents(student_id, query, n_results=3)
            documents = search_results['documents'][0] if search_results['documents'] else []
            metadatas = search_results['metadatas'][0] if search_results['metadatas'] else []
            
            print(f"Retrieved {len(documents)} documents")
            
            if not documents:
                analysis_prompt = f"Student {student_id} asked: {query}. They have no performance data yet. Provide general study advice."
                advice_prompt = f"Student asked: {query}. They haven't added any performance data. Provide helpful general advice for students."
                
                analysis = self.call_groq_api(analysis_prompt)
                advice = self.call_groq_api(advice_prompt)
                
                return {
                    "response": advice,
                    "analysis": analysis,
                    "documents_used": 0
                }
            
            doc_content = "\n\n".join(documents)
            print(f"Document content: {doc_content}")
            
            analysis_prompt = f"Analyze this student performance data: {doc_content}"
            analysis = self.call_groq_api(analysis_prompt)
            print(f"Analysis result: {analysis}")
            
            advice_prompt = f"Student asked: {query}. Performance data: {doc_content}. Analysis: {analysis}. Provide specific advice."
            advice = self.call_groq_api(advice_prompt)
            print(f"Advice result: {advice}")
            
            return {
                "response": advice,
                "analysis": analysis,
                "documents_used": len(documents)
            }
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "response": f"System error: {str(e)}",
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
            "response": f"Request failed: {str(e)}",
            "analysis": f"Error: {str(e)}",
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
