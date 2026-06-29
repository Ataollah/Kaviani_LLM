import base64
import httpx
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def get_openrouter_client():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key='your_key_here',
        base_url="https://openrouter.ai/api/v1"
    )

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_mime_type(image_path: str) -> str:
    extension = image_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return mime_types.get(extension, 'image/jpeg')

def extract_national_code_with_langchain(image_path: str) -> tuple:
    
    try:
        llm = get_openrouter_client()
        
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        messages = [
            SystemMessage(content="""شما یک سیستم OCR هستید که اطلاعات کارت ملی ایرانی را استخراج می‌کنید.
فقط کد ملی ۱۰ رقمی را استخراج کنید.
پاسخ را فقط به صورت یک عدد ۱۰ رقمی بدهید.
اگر کد ملی پیدا نشد، کلمه "NOT_FOUND" را برگردانید.
                          کد ملی را به صورت عدد و فقط به صورت انگلیسی را برگردان. یعنی اعداد
                          ۱۲۷۰۷۶۵۱۰۸ 
                          نباشد. یعنی به صورت
                          1270765108
                          باشد.
                          """),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "لطفاً کد ملی را از این تصویر کارت ملی استخراج کنید. فقط عدد ۱۰ رقمی کد ملی را برگردانید."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ])
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip()
        
        numbers = re.findall(r'\d+', result)
        for num in numbers:
            if len(num) == 10:
                return num, None
        
        if "NOT_FOUND" in result:
            return None, "کد ملی در تصویر یافت نشد"
        
        return None, "کد ملی معتبر در تصویر یافت نشد"
        
    except Exception as e:
        return None, f"خطا در پردازش تصویر: {str(e)}"

def extract_national_code_with_httpx(image_path: str, api_key: str, model: str = "openai/gpt-4o") -> tuple:
    """
    Extracts the 10-digit Iranian national code from an image using OpenRouter API.

    This function uses a role-playing prompt that instructs the model to:
    - Act as an expert OCR assistant specialized in Iranian national ID cards.
    - Verify that the image indeed contains a valid Iranian national ID card.
    - If the image is unrelated (e.g., a passport, driver's license, or any other document),
      respond with the exact string "IMAGE_NOT_RELATED".
    - Otherwise, extract and return only the 10-digit national code (English digits).

    Args:
        image_path (str): Path to the image file.
        api_key (str): OpenRouter API key.
        model (str): Model identifier (default: "openai/gpt-4o").

    Returns:
        tuple: (national_code or None, error_message or None)
    """
    if not api_key:
        return None, "کلید API وارد نشده است"

    try:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Tax Calculator App",
            "Content-Type": "application/json"
        }

        # Enhanced system prompt with role-playing and unrelated image handling
        system_prompt = """
شما یک دستیار هوشمند و متخصص در تشخیص و استخراج اطلاعات از کارت‌های ملی ایران هستید.
شما باید ابتدا بررسی کنید که تصویر ارسال‌شده، یک کارت ملی معتبر ایرانی است یا خیر.

اگر تصویر کارت ملی نباشد (مثلاً پاسپورت، گواهینامه، کارت بانکی، یا هر سند دیگری)،
دقیقاً عبارت "IMAGE_NOT_RELATED" را به عنوان پاسخ برگردانید.

اگر تصویر کارت ملی است، تنها کد ملی ۱۰ رقمی آن را استخراج کنید.
کد ملی را فقط به صورت عدد انگلیسی (مثلاً ۱۲۷۰۷۶۵۱۰۸) برگردانید و نه به صورت فارسی (۱۲۷۰۷۶۵۱۰۸).
پاسخ شما باید دقیقاً همان عدد ۱۰ رقمی باشد و هیچ توضیح اضافی نداشته باشد.

نکته: اگر کد ملی در تصویر وجود ندارد، عبارت "NOT_FOUND" را برگردانید.
"""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "لطفاً کد ملی را از این تصویر استخراج کنید. اگر تصویر کارت ملی نیست، عبارت IMAGE_NOT_RELATED را برگردانید."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 50
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

        result_json = response.json()
        result = result_json["choices"][0]["message"]["content"].strip()

        # Check for special tokens first
        if "IMAGE_NOT_RELATED" in result:
            return None, "تصویر ارسال‌شده یک کارت ملی معتبر نیست"

        if "NOT_FOUND" in result:
            return None, "کد ملی در تصویر یافت نشد"

        # Extract all numeric sequences
        numbers = re.findall(r'\d+', result)
        for num in numbers:
            if len(num) == 10:
                return num, None

        # If we reach here, no valid 10-digit code was found
        return None, "کد ملی معتبر در تصویر یافت نشد"

    except httpx.HTTPStatusError as e:
        return None, f"خطای HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return None, f"خطا در پردازش تصویر: {str(e)}"

def extract_national_code_from_image(image_path: str) -> tuple:
    return extract_national_code_with_langchain(image_path)

def validate_national_code(national_code: str) -> bool:
    if not national_code or len(national_code) != 10:
        return False
    
    if not national_code.isdigit():
        return False
    
    if national_code == national_code[0] * 10:
        return False
    
    try:
        check = int(national_code[9])
        total = sum(int(national_code[i]) * (10 - i) for i in range(9))
        remainder = total % 11
        
        if remainder < 2:
            return check == remainder
        else:
            return check == (11 - remainder)
    except:
        return False

AVAILABLE_MODELS = [
    ("openai/gpt-4o", "GPT-4o (OpenAI)"),
    ("openai/gpt-4o-mini", "GPT-4o Mini (OpenAI)"),
    ("openai/gpt-4-turbo", "GPT-4 Turbo (OpenAI)"),
    ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (Anthropic)"),
    ("anthropic/claude-3-opus", "Claude 3 Opus (Anthropic)"),
    ("google/gemini-pro-vision", "Gemini Pro Vision (Google)"),
    ("google/gemini-pro-1.5", "Gemini 1.5 Pro (Google)"),
    ("meta-llama/llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision (Meta)"),
]

def get_model_choices():
    return [model[0] for model in AVAILABLE_MODELS]

def get_model_labels():
    return {model[0]: model[1] for model in AVAILABLE_MODELS}