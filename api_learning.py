from fastapi import FastAPI , HTTPException
from pydantic import BaseModel , Field


app = FastAPI()


transactions = {
    1: {"id": 1, "amount": 120, "payment_type": 4},
    2: {"id": 2, "amount": 350, "payment_type": 2},
}

class TransactionIn(BaseModel):
    transaction_id: int
    amount: int
class TransactionOut(BaseModel):
    id : int
    transaction_id: int
    amount: int
class TransactionPatch(BaseModel):
    amount: float | None = None
    payment_type: int | None = None


@app.get("/transactions")
def read_transactions():
    return list(transactions.values())

@app.get("/transactions/{transactions_id}")
def read_transaction(transactions_id:int ):
    if transactions_id not  in transactions:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transactions[transactions_id]

@app.post("/transactions" , response_model=TransactionOut , status_code=201)
def create_transaction(transaction: TransactionIn):
    new_id= max(transactions.keys())+1
    transactions[new_id] = {
        "id": new_id,
        **transaction.model_dump()
    }
    return transaction
