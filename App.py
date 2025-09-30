import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
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
            "tags": question_data.tags,
            "date": question_data.date,
            "feedback": question_data.feedback
        }
        doc = Document(page_content=doc_text, metadata=metadata)
        self.vector_store.add_documents([doc])
        try:
            self.vector_store.persist()
        except Exception:
            pass

    def get_retriever(self):
        self._ensure_vector_store()
        return self.vector_store.as_retriever()

vector_manager = VectorStoreManager()

class IntentAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": f"Analyze the student's query and determine the intent and relevant filters. Query: {query}"}],
                model="llama3-70b-8192"
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception:
            return {"intent": "performance_review", "subject_filter": None, "topic_filter": None}

class PerformanceAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_performance(self, documents: List[Document]) -> Dict[str, Any]:
        doc_texts = [doc.page_content for doc in documents]
        context = "\n\n".join(doc_texts)
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": f"Analyze the student's performance data and identify strengths and weaknesses. Performance Data: {context}"}],
            model="llama3-70b-8192"
        )
        return {"analysis": chat_completion.choices[0].message.content, "documents": documents}

class Advisor:
    def __init__(self):
        self.client = client

    def generate_advice(self, analysis: Dict[str, Any], intent: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Based on the performance analysis and student intent, generate actionable advice. Intent: {intent} Performance Analysis: {analysis['analysis']}"
            }],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content

class StudentChatbot:
    def __init__(self):
        self.intent_analyzer = IntentAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.advisor = Advisor()
        self.vector_manager = vector_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = Graph()
        workflow.add_node("intent_analyzer", self._intent_analyzer_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("performance_analyzer", self._performance_analyzer_node)
        workflow.add_node("advisor", self._advisor_node)
        workflow.add_node("response", self._response_node)
        workflow.set_entry_point("intent_analyzer")
        workflow.add_edge("intent_analyzer", "retriever")
        workflow.add_edge("retriever", "performance_analyzer")
        workflow.add_edge("performance_analyzer", "advisor")
        workflow.add_edge("advisor", "response")
        workflow.add_edge("response", END)
        return workflow.compile()

    def _intent_analyzer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent_data = self.intent_analyzer.analyze_intent(state["query"])
        return {"intent": intent_data}

    def _retriever_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        retriever = self.vector_manager.get_retriever()
        filters = {"student_id": state["student_id"]}
        if state["intent"].get("subject_filter"):
            filters["tags"] = state["intent"]["subject_filter"]
        documents = retriever.get_relevant_documents(
            state["query"],
            filter=filters
        )
        return {"documents": documents}

    def _performance_analyzer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        analysis = self.performance_analyzer.analyze_performance(state["documents"])
        return {"performance_analysis": analysis}

    def _advisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        advice = self.advisor.generate_advice(
            state["performance_analysis"],
            state["intent"].get("intent", "performance_review")
        )
        return {"advice": advice}

    def _response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": state["advice"],
            "analysis": state["performance_analysis"]["analysis"],
            "documents_used": len(state["documents"])
        }

    def query(self, student_id: str, query: str) -> Dict[str, Any]:
        initial_state = {
            "student_id": student_id,
            "query": query
        }
        result = self.graph.invoke(initial_state)
        return result

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
        result = chatbot.query(chat_query.student_id, chat_query.query)
        return {
            "response": result["response"],
            "analysis": result["analysis"],
            "documents_used": result["documents_used"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
