from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

from ocr_service import extract_national_code_from_image, validate_national_code
from database import find_citizen_by_national_code
from tax_service import calculate_tax, generate_tax_report

class TaxState(TypedDict):
    messages: Annotated[list, operator.add]
    image_path: str
    api_key: str
    model: str
    national_code: str
    citizen_data: dict
    tax_result: dict
    error: str
    current_step: str

def welcome_node(state: TaxState) -> dict:
    welcome_message = """
سلام! 👋
به سیستم محاسبه مالیات خوش آمدید.

برای محاسبه مالیات، لطفاً تصویر کارت ملی خود را آپلود کنید.
"""
    return {
        "messages": [AIMessage(content=welcome_message)],
        "current_step": "waiting_for_image"
    }

def process_image_node(state: TaxState) -> dict:
    image_path = state.get("image_path")
    
    if not image_path:
        return {
            "messages": [AIMessage(content="❌ لطفاً تصویر کارت ملی را آپلود کنید.")],
            "error": "تصویر آپلود نشده",
            "current_step": "waiting_for_image"
        }
    
    model = state.get("model", "openai/gpt-4o")
    
    return {
        "messages": [AIMessage(content=f"🔄 در حال پردازش تصویر کارت ملی با مدل {model}...")],
        "current_step": "extracting_national_code"
    }

def extract_national_code_node(state: TaxState) -> dict:
    image_path = state.get("image_path")
    
    national_code, error = extract_national_code_from_image(image_path)
    
    if error:
        return {
            "messages": [AIMessage(content=f"❌ خطا: {error}")],
            "error": error,
            "current_step": "error"
        }
    
    if not validate_national_code(national_code):
        return {
            "messages": [AIMessage(content=f"❌ کد ملی استخراج شده ({national_code}) معتبر نیست.")],
            "error": "کد ملی نامعتبر",
            "current_step": "error"
        }
    
    return {
        "messages": [AIMessage(content=f"✅ کد ملی استخراج شد: {national_code}")],
        "national_code": national_code,
        "current_step": "searching_database"
    }

def search_database_node(state: TaxState) -> dict:
    national_code = state.get("national_code")
    
    if not national_code:
        return {
            "messages": [AIMessage(content="❌ کد ملی یافت نشد.")],
            "error": "کد ملی یافت نشد",
            "current_step": "error"
        }
    
    citizen_data = find_citizen_by_national_code(national_code)
    
    if not citizen_data:
        return {
            "messages": [AIMessage(content=f"❌ کاربری با کد ملی {national_code} در سیستم ثبت نشده است.")],
            "error": "کاربر یافت نشد",
            "current_step": "not_found"
        }
    
    if not citizen_data.get('is_active'):
        return {
            "messages": [AIMessage(content="❌ حساب کاربری غیرفعال است.")],
            "error": "حساب غیرفعال",
            "current_step": "error"
        }
    
    return {
        "messages": [AIMessage(content=f"✅ کاربر یافت شد: {citizen_data['first_name']} {citizen_data['last_name']}")],
        "citizen_data": citizen_data,
        "current_step": "calculating_tax"
    }

def calculate_tax_node(state: TaxState) -> dict:
    citizen_data = state.get("citizen_data")
    
    if not citizen_data:
        return {
            "messages": [AIMessage(content="❌ اطلاعات کاربر یافت نشد.")],
            "error": "اطلاعات کاربر یافت نشد",
            "current_step": "error"
        }
    
    tax_result = calculate_tax(citizen_data)
    report = generate_tax_report(citizen_data, tax_result)
    
    return {
        "messages": [AIMessage(content=report)],
        "tax_result": tax_result,
        "current_step": "completed"
    }

def error_node(state: TaxState) -> dict:
    error = state.get("error", "خطای نامشخص")
    return {
        "messages": [AIMessage(content=f"""
❌ متأسفانه خطایی رخ داد: {error}

لطفاً دوباره تلاش کنید یا با پشتیبانی تماس بگیرید.
""")],
        "current_step": "error"
    }

def router(state: TaxState) -> Literal["process_image", "extract_code", "search_db", "calculate", "error", "end"]:
    step = state.get("current_step", "start")
    
    if step == "waiting_for_image":
        if state.get("image_path"):
            return "process_image"
        return "end"
    elif step == "extracting_national_code":
        return "extract_code"
    elif step == "searching_database":
        return "search_db"
    elif step == "calculating_tax":
        return "calculate"
    elif step in ["error", "not_found"]:
        return "error"
    elif step == "completed":
        return "end"
    
    return "end"

def create_tax_graph():
    workflow = StateGraph(TaxState)
    
    workflow.add_node("welcome", welcome_node)
    workflow.add_node("process_image", process_image_node)
    workflow.add_node("extract_code", extract_national_code_node)
    workflow.add_node("search_db", search_database_node)
    workflow.add_node("calculate", calculate_tax_node)
    workflow.add_node("error", error_node)
    
    workflow.set_entry_point("welcome")
    
    workflow.add_conditional_edges(
        "welcome",
        router,
        {
            "process_image": "process_image",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "process_image",
        router,
        {
            "extract_code": "extract_code",
            "error": "error",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "extract_code",
        router,
        {
            "search_db": "search_db",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "search_db",
        router,
        {
            "calculate": "calculate",
            "error": "error"
        }
    )
    
    workflow.add_edge("calculate", END)
    workflow.add_edge("error", END)
    
    return workflow.compile()

tax_graph = create_tax_graph()

def run_tax_calculation(image_path: str, api_key: str = '', model: str = "openai/gpt-4o") -> str:
    initial_state = {
        "messages": [],
        "image_path": image_path,
        "api_key": api_key,
        "model": model,
        "national_code": "",
        "citizen_data": {},
        "tax_result": {},
        "error": "",
        "current_step": "start"
    }
    
    result = tax_graph.invoke(initial_state)
    
    all_messages = []
    for msg in result.get("messages", []):
        if hasattr(msg, 'content'):
            all_messages.append(msg.content)
    
    return "\n".join(all_messages)