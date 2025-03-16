from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Initialize FastAPI app
app = FastAPI()

# Download Sentiment Intensity Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/polling_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class PollDB(Base):
    __tablename__ = "polls"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)

class ResponseDB(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, index=True)
    answer = Column(Text)
    sentiment = Column(String)

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Poll Model
class Poll(BaseModel):
    question: str

# Response Model
class Response(BaseModel):
    poll_id: int
    answer: str

@app.post("/create_poll/")
def create_poll(poll: Poll, db: Session = Depends(get_db)):
    new_poll = PollDB(question=poll.question)
    db.add(new_poll)
    db.commit()
    db.refresh(new_poll)
    return {"message": "Poll created successfully", "poll": new_poll}

@app.get("/get_polls/")
def get_polls(db: Session = Depends(get_db)):
    polls = db.query(PollDB).all()
    return polls

@app.post("/submit_response/")
def submit_response(response: Response, db: Session = Depends(get_db)):
    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(response.answer)
    sentiment = "positive" if sentiment_score['compound'] > 0.05 else "negative" if sentiment_score['compound'] < -0.05 else "neutral"
    
    new_response = ResponseDB(poll_id=response.poll_id, answer=response.answer, sentiment=sentiment)
    db.add(new_response)
    db.commit()
    return {"message": "Response submitted successfully", "response": new_response}

@app.get("/get_results/{poll_id}")
def get_results(poll_id: int, db: Session = Depends(get_db)):
    poll_responses = db.query(ResponseDB).filter(ResponseDB.poll_id == poll_id).all()
    if not poll_responses:
        raise HTTPException(status_code=404, detail="No responses found for this poll")
    
    df = pd.DataFrame([{ "sentiment": r.sentiment } for r in poll_responses])
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    
    return {"poll_id": poll_id, "sentiment_distribution": sentiment_counts, "responses": poll_responses}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
